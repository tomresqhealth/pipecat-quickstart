# bot.py
import os
import sys
import asyncio
import json
import time
import urllib.request
import re
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from pipecat.frames.frames import (
    LLMMessagesAppendFrame, 
    TextFrame, 
    LLMFullResponseEndFrame
)
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
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection 

from pipecat.transports.daily.transport import (
    DailyTransport, 
    DailyParams, 
    DailyOutputTransportMessageFrame 
)

from prompts import SYSTEM_PROMPT, GREETING_TEXT, VISUAL_INSTRUCTIONS
from images import get_image_url
from videos import get_video_url 

logger.remove()
logger.add(sys.stderr, level="DEBUG")

# -------------------------------------------------------------------------
# INTENT FALLBACK PROCESSOR (The "Eavesdropper")
# -------------------------------------------------------------------------
class IntentFallbackProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        # UPDATED REGEX: More robust patterns
        # \s* = optional whitespace
        # (?:the\s+)? = optional word "the"
        # We also added "restart" as a distinct intent
        self.intents = {
            "pause": re.compile(r"\bpausing\s*(?:the\s+)?video\b", re.IGNORECASE),
            "play": re.compile(r"\bresuming\s*(?:the\s+)?video\b", re.IGNORECASE),
            "mute": re.compile(r"\bmuting\s*(?:the\s+)?video\b", re.IGNORECASE),
            "unmute": re.compile(r"\bunmuting\s*(?:the\s+)?video\b", re.IGNORECASE),
            "skip": re.compile(r"\bskipping\s*forward\b", re.IGNORECASE),
            "rewind": re.compile(r"\brewinding\s*(?:the\s+)?video\b", re.IGNORECASE),
            "restart": re.compile(r"\brestarting\s*(?:the\s+)?video\b", re.IGNORECASE)
        }
        self._triggered_intents = set()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseEndFrame):
            self._triggered_intents.clear()

        if isinstance(frame, TextFrame):
            text = frame.text
            
            for action, pattern in self.intents.items():
                if action not in self._triggered_intents and pattern.search(text):
                    logger.info(f"🕵️ Eavesdropper detected STRICT intent: '{action}' in text: '{text}'")
                    
                    msg = DailyOutputTransportMessageFrame(message={
                        "event": "video_control", 
                        "command": action,
                        "source": "fallback_processor"
                    })
                    await self.push_frame(msg, direction)
                    self._triggered_intents.add(action)
                    # No break, allows multiple commands in one turn

        await self.push_frame(frame, direction)

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
                "name": "show_video",
                "description": "Display and play a video. Use 'skiing' if the user asks for a generic video or skiing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "The specific video key to show (e.g. 'skiing'). Defaults to 'skiing' if unspecified."
                        }
                    },
                    "required": ["keyword"]
                }
            },
            {
                "name": "close_image",
                "description": "Close the currently displayed image or video.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                }
            },
            {
                "name": "control_video",
                "description": "Control video playback (pause, play, mute, unmute, skip, rewind, restart).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["pause", "play", "mute", "unmute", "skip", "rewind", "restart"]
                        }
                    },
                    "required": ["action"]
                }
            }
        ]
    }
]

async def bot(args: RunnerArguments):
    print(f"\n{'='*40}")
    print(f"🤖 QUADRIGA BOT (Robust Control)")
    print(f"{'='*40}\n")

    transport = None
    greeting_triggered = False

    # --- TRANSPORT SETUP ---
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            start_secs=0.2,      
            stop_secs=0.8,       
            confidence=0.7,      
            min_volume=0.6       
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
            safety_settings=safety_settings
        )
    )

    # --- PIPELINE SETUP ---
    messages = []
    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)
    rtvi = RTVIProcessor(config=RTVIConfig(config=[], enable_bot_ready_message=True))
    
    fallback_processor = IntentFallbackProcessor()

    pipeline = Pipeline([
        transport.input(), 
        rtvi, 
        user_aggregator, 
        llm, 
        fallback_processor, 
        transport.output(), 
        assistant_aggregator
    ])

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))

    # --- HANDLERS ---
    async def show_image_handler(params: FunctionCallParams):
        keyword = params.arguments.get("keyword", "").lower()
        image_url = get_image_url(keyword)
        logger.info(f"🎨 Tool Triggered: show_image ({keyword})")
        
        try:
            frame = DailyOutputTransportMessageFrame(
                message={"event": "show_image", "url": image_url}
            )
            await transport.output().process_frame(frame, FrameDirection.DOWNSTREAM)
        except Exception as e:
            logger.error(f"❌ Failed to send frame: {e}")
        
        return f"Displaying image for {keyword}."

    async def show_video_handler(params: FunctionCallParams):
        keyword = params.arguments.get("keyword", "skiing").lower()
        video_url = get_video_url(keyword)
        
        if not video_url:
            logger.warning(f"⚠️ Video keyword '{keyword}' not found. Defaulting to skiing.")
            video_url = get_video_url("skiing")

        logger.info(f"🎬 Tool Triggered: show_video ({keyword}) -> {video_url}")
        try:
            frame = DailyOutputTransportMessageFrame(
                message={"event": "show_video", "url": video_url}
            )
            await transport.output().process_frame(frame, FrameDirection.DOWNSTREAM)
        except Exception as e:
            logger.error(f"❌ Failed to send frame: {e}")
        return f"Video displayed on screen."

    async def close_image_handler(params: FunctionCallParams):
        logger.info("❌ Tool Triggered: close_image")
        try:
            frame = DailyOutputTransportMessageFrame(
                message={"event": "close_image"}
            )
            await transport.output().process_frame(frame, FrameDirection.DOWNSTREAM)
        except Exception as e:
            logger.error(f"❌ Failed to send frame: {e}")
        return "Media closed."

    async def control_video_handler(params: FunctionCallParams):
        action = params.arguments.get("action")
        logger.info(f"🎬 Tool Triggered: control_video -> {action}")
        try:
            frame = DailyOutputTransportMessageFrame(
                message={
                    "event": "video_control", 
                    "command": action,
                    "source": "tool_call"
                }
            )
            await transport.output().process_frame(frame, FrameDirection.DOWNSTREAM)
        except Exception as e:
            logger.error(f"❌ Failed to send video control frame: {e}")
        return f"Video {action} command sent."

    # REGISTER FUNCTIONS
    llm.register_function("show_image", show_image_handler, cancel_on_interruption=False)
    llm.register_function("show_video", show_video_handler, cancel_on_interruption=False)
    llm.register_function("close_image", close_image_handler, cancel_on_interruption=False)
    llm.register_function("control_video", control_video_handler, cancel_on_interruption=False)

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

def get_daily_token(room_url, api_key):
    if not api_key:
        logger.error("❌ DAILY_API_KEY is missing in .env!")
        return None
    try:
        room_name = room_url.strip('/').split('/')[-1]
        url = "https://api.daily.co/v1/meeting-tokens"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        expiration = int(time.time() + 3600) 
        data = { "properties": { "room_name": room_name, "is_owner": True, "exp": expiration } }
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
        args = DailyRunnerArguments(room_url=ROOM_URL, token=meeting_token)
        try:
            asyncio.run(bot(args))
        except KeyboardInterrupt:
            pass
    else:
        logger.error("🛑 Cannot start bot without a valid token.")