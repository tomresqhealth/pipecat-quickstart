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
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.runner.types import DailyRunnerArguments, SmallWebRTCRunnerArguments, RunnerArguments
from pipecat.services.llm_service import FunctionCallParams
from pipecat.processors.frame_processor import FrameDirection 

from pipecat.transports.daily.transport import (
    DailyTransport, 
    DailyParams, 
    DailyOutputTransportMessageFrame 
)

# --- IMPORTS FROM MODULES ---
from prompts import SYSTEM_PROMPT, GREETING_TEXT, VISUAL_INSTRUCTIONS
from images import get_image_url

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
                "description": "Display an image to the user based on a specific keyword.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "The specific item key to show (e.g. 'birthday', 'headshot', 'fenway_park')."
                        }
                    },
                    "required": ["keyword"]
                }
            },
            {
                "name": "close_image",
                "description": "Close the currently displayed image.",
                "parameters": {
                    "type": "object",
                    "properties": {},
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
    # Stricter VAD settings to prevent background noise interruptions
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            start_secs=0.5,      
            stop_secs=1.2,       # Slightly reduced from 1.5 for better responsiveness
            confidence=0.8,      
            min_volume=0.8       
        )
    )

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
                vad_analyzer=vad_analyzer
            )
        )

    elif isinstance(args, SmallWebRTCRunnerArguments):
        from pipecat.transports.base_transport import TransportParams
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
        logger.info(f"💻 Mode: SmallWebRTC (Local)")
        
        transport = SmallWebRTCTransport(
            webrtc_connection=args.webrtc_connection,
            params=TransportParams(
                audio_out_enabled=True, 
                audio_in_enabled=True, 
                vad_analyzer=vad_analyzer
            )
        )

    # --- LLM SETUP ---
    combined_system_prompt = SYSTEM_PROMPT + VISUAL_INSTRUCTIONS

    # Define safety settings to prevent "Policy Violation" disconnects
    # This fixes the 1008 error that killed your previous session
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

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
            },
            safety_settings=safety_settings # <--- APPLIED HERE
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

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))

    # --- HANDLERS ---
    async def show_image_handler(params: FunctionCallParams):
        function_name = params.function_name
        args = params.arguments
        
        logger.info(f"🎨 Tool Triggered: {function_name} with {args}")
        
        keyword = args.get("keyword", "").lower()
        image_url = get_image_url(keyword)

        try:
            frame = DailyOutputTransportMessageFrame(
                message={"event": "show_image", "url": image_url}
            )
            # FIX: Send directly to transport output to bypass LLM blocking
            await transport.output().process_frame(frame, FrameDirection.DOWNSTREAM)
            logger.info(f"📡 Sent App Message Frame: {image_url}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send frame: {e}")
        
        return f"Displaying image for {keyword}."

    async def close_image_handler(params: FunctionCallParams):
        logger.info("❌ Tool Triggered: close_image")
        try:
            frame = DailyOutputTransportMessageFrame(
                message={"event": "close_image"}
            )
            # FIX: Send directly to transport output to bypass LLM blocking
            await transport.output().process_frame(frame, FrameDirection.DOWNSTREAM)
            logger.info("📡 Sent App Message Frame: close_image")
        except Exception as e:
            logger.error(f"❌ Failed to send frame: {e}")
        return "Image closed."

    # REGISTER FUNCTIONS
    llm.register_function("show_image", show_image_handler, cancel_on_interruption=False)
    llm.register_function("close_image", close_image_handler, cancel_on_interruption=False)

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