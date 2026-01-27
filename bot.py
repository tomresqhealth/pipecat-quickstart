# bot.py
import os
import sys
import asyncio
import json
import time
import urllib.request
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIConfig
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, InputParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.runner.types import DailyRunnerArguments, SmallWebRTCRunnerArguments, RunnerArguments
from pipecat.services.llm_service import FunctionCallParams # <--- NEW IMPORT

from pipecat.transports.daily.transport import (
    DailyTransport, 
    DailyParams, 
    DailyOutputTransportMessageFrame 
)

from prompts import SYSTEM_PROMPT, GREETING_TEXT

logger.remove()
logger.add(sys.stderr, level="DEBUG")

# -------------------------------------------------------------------------
# TOOL DEFINITIONS
# -------------------------------------------------------------------------
tools = [
    {
        "function_declarations": [
            {
                "name": "show_image",
                "description": "Display an image to the user. Use this specifically when the user asks to see a person or an object.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "The specific item to show. Options: 'tom_elliott'."
                        }
                    },
                    "required": ["keyword"]
                }
            }
        ]
    }
]

async def bot(args: RunnerArguments):
    print(f"\n{'='*40}")
    print(f"🤖 QUADRIGA BOT (Profile Mode)")
    print(f"{'='*40}\n")

    transport = None
    greeting_triggered = False

    # --- TRANSPORT SETUP ---
    if isinstance(args, DailyRunnerArguments):
        logger.info(f"📹 Mode: Daily (Cloud/URL) -> {args.room_url}")
        
        transport = DailyTransport(
            room_url=args.room_url,
            token=args.token,
            bot_name="Quadriga",
            params=DailyParams(
                audio_out_enabled=True,
                transcription_enabled=False,
                audio_in_enabled=True,
                # vad_enabled=True, <--- REMOVED (Deprecated)
                vad_analyzer=SileroVADAnalyzer()
            )
        )

    elif isinstance(args, SmallWebRTCRunnerArguments):
        from pipecat.transports.base_transport import TransportParams
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
        logger.info(f"💻 Mode: SmallWebRTC (Local)")
        
        transport = SmallWebRTCTransport(
            webrtc_connection=args.webrtc_connection,
            params=TransportParams(audio_out_enabled=True, audio_in_enabled=True, vad_analyzer=SileroVADAnalyzer())
        )

    # --- LLM SETUP ---
    VISUAL_INSTRUCTIONS = """
    # VISUAL AID INSTRUCTIONS
    You have a tool called 'show_image'.
    - If the user asks "Can you show me a picture of Tom Elliott?" -> Call 'show_image' with keyword 'tom_elliott'.
    - If the user asks "Who is Tom?" or "What does Tom look like?" -> Call the tool.
    """
    
    combined_system_prompt = SYSTEM_PROMPT + VISUAL_INSTRUCTIONS

    llm = GeminiLiveLLMService(
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        model="models/gemini-2.5-flash-native-audio-preview-12-2025",
        voice_id="Charon", 
        transcribe_user_audio=True,
        system_instruction=combined_system_prompt,
        tools=tools,
        params=InputParams(
            generation_config={
                "response_modalities": ["AUDIO"],
                "temperature": 0.0,
            }
        )
    )

    # --- PIPELINE SETUP ---
    messages = []
    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)
    rtvi = RTVIProcessor(config=RTVIConfig(config=[], enable_bot_ready_message=True))

    pipeline = Pipeline([
        transport.input(), rtvi, user_aggregator, llm, transport.output(), assistant_aggregator
    ])

    # 1. CREATE TASK FIRST
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))

    # 2. DEFINE HANDLER (Now it can see `task`)
    async def show_image_handler(params: FunctionCallParams):
        # We extract args from the new params object
        function_name = params.function_name
        args = params.arguments
        
        logger.info(f"🎨 Tool Triggered: {function_name} with {args}")
        
        image_db = {
            "tom_elliott": "https://nutum.ai/wp-content/uploads/2025/03/tom-headshot-2024.jpeg"
        }

        keyword = args.get("keyword", "").lower()
        image_url = image_db.get(keyword, "https://placehold.co/600x400?text=Image+Not+Found")

        try:
            # Create the frame
            frame = DailyOutputTransportMessageFrame(
                message={"event": "show_image", "url": image_url}
            )
            
            # THE FIX: Queue the frame on the pipeline task
            # This pushes the frame into the stream, which flows to transport.output()
            await task.queue_frames([frame])
            
            logger.info(f"📡 Sent App Message Frame: {image_url}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send frame: {e}")
        
        # Note: You don't need to call result_callback manually anymore if you return the result string
        # However, to maintain the conversation flow properly via the LLM service:
        return f"Displaying image of {keyword}."

    # 3. REGISTER FUNCTION
    llm.register_function("show_image", show_image_handler)

    # --- GREETING & RUNNER ---
    async def trigger_greeting():
        nonlocal greeting_triggered
        if greeting_triggered: return
        greeting_triggered = True
        
        await task.queue_frames([
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": f"Please say exactly this text: {GREETING_TEXT}"}],
                run_llm=True
            )
        ])

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        await trigger_greeting()

    if isinstance(args, DailyRunnerArguments):
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await trigger_greeting()

    runner = PipelineRunner()
    await runner.run(task)

# --- HELPER: AUTO-GENERATE TOKEN ---
def get_daily_token(room_url, api_key):
    if not api_key:
        logger.error("❌ DAILY_API_KEY is missing in .env!")
        return None
        
    try:
        room_name = room_url.strip('/').split('/')[-1]
        logger.info(f"🎫 Generating token for room: {room_name}")
        
        url = "https://api.daily.co/v1/meeting-tokens"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        expiration = int(time.time() + 3600) 
        
        data = {
            "properties": {
                "room_name": room_name,
                "is_owner": True,
                "exp": expiration
            }
        }
        
        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read())
            return res["token"]
            
    except Exception as e:
        logger.error(f"❌ Failed to generate token: {e}")
        return None

if __name__ == "__main__":
    from pipecat.runner.types import DailyRunnerArguments
    
    ROOM_URL = "https://resqhealth.daily.co/quadriga-agent-test"
    API_KEY = os.getenv("DAILY_API_KEY", "")

    meeting_token = get_daily_token(ROOM_URL, API_KEY)
    
    if meeting_token:
        logger.info("🚀 FORCING DAILY MODE...")
        args = DailyRunnerArguments(
            room_url=ROOM_URL,
            token=meeting_token 
        )
        try:
            asyncio.run(bot(args))
        except KeyboardInterrupt:
            pass
    else:
        logger.error("🛑 Cannot start bot without a valid token.")