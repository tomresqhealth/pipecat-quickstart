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
    LLMFullResponseEndFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    EndFrame,
    ErrorFrame,
    StartFrame,
    CancelFrame
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIConfig

# MODERN v0.0.101 AGGREGATORS
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator, 
    LLMUserAggregatorParams,
    LLMContextAggregatorPair
)

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

# Restore original full logging configuration for deep debugging
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="DEBUG")

# -------------------------------------------------------------------------
# 1. CLIENT STATE NOTIFIER (Audio Ducking & Border Control)
# -------------------------------------------------------------------------
class ClientStateNotifier(FrameProcessor):
    def __init__(self):
        super().__init__()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        # Notify frontend of user speaking status for UI/Audio Ducking logic in index.html
        if isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("User started speaking - Notifying Client for ducking")
            msg = DailyOutputTransportMessageFrame(message={
                "event": "user_speaking_status", 
                "status": "speaking"
            })
            await self.push_frame(msg, direction)
            
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("User stopped speaking - Notifying Client for restore")
            msg = DailyOutputTransportMessageFrame(message={
                "event": "user_speaking_status", 
                "status": "stopped"
            })
            await self.push_frame(msg, direction)

        await self.push_frame(frame, direction)

# -------------------------------------------------------------------------
# 2. PANIC BUTTON PROCESSOR (Emergency Low-Latency Stops)
# -------------------------------------------------------------------------
class PanicButtonProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        
        # Matches emergency pause keywords instantly before LLM finishes thinking
        self.intent_patterns = [
            (re.compile(r"\b(?:stop|pause|hold on|wait|shut up|freeze)\b", re.IGNORECASE), "pause"),
        ]
        self._triggered_intents = set()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseEndFrame):
            self._triggered_intents.clear()

        if isinstance(frame, TextFrame):
            text = frame.text
            for pattern, command in self.intent_patterns:
                if command not in self._triggered_intents and pattern.search(text):
                    logger.info(f"🚨 Panic Button Triggered: Detected '{command}' in user speech.")
                    msg_payload = {
                        "event": "video_control", 
                        "command": command, 
                        "source": "panic_button_fallback",
                        "timestamp": time.time()
                    }
                    await self.push_frame(DailyOutputTransportMessageFrame(message=msg_payload), direction)
                    self._triggered_intents.add(command)

        await self.push_frame(frame, direction)

# -------------------------------------------------------------------------
# ATOMIC TOOL DEFINITIONS (Precision Volume, Targeted Zoom & Pan)
# -------------------------------------------------------------------------
tools = [
    {
        "function_declarations": [
            {
                "name": "show_image",
                "description": "Display a photo with precision zoom and coordinate-based panning.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": { "type": "string", "description": "Image keyword from the database." },
                        "zoom": { "type": "number", "description": "Scale factor. 1.0 is normal, 10.0 is extreme detail zoom." },
                        "pan_x": { "type": "number", "description": "Horizontal focus point as % (0 to 100). 50 is center." },
                        "pan_y": { "type": "number", "description": "Vertical focus point as % (0 to 100). 50 is center." }
                    },
                    "required": ["keyword"]
                }
            },
            {
                "name": "update_video_state",
                "description": "Atomic tool to control entire video state, including precision volume and targeted zoom.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "playing": { "type": "boolean", "description": "True to play, False to pause." },
                        "muted": { "type": "boolean", "description": "True to mute audio." },
                        "volume": { "type": "number", "description": "Precision volume level (0.0 to 1.0)." },
                        "time_abs": { "type": "number", "description": "Jump to an absolute second (0 to restart)." },
                        "time_rel": { "type": "number", "description": "Seek seconds relative to current time (+/-)." },
                        "speed": { "type": "number", "description": "Playback rate (0.5 to 2.0)." },
                        "zoom": { "type": "number", "description": "Video scale factor (1.0 is normal)." },
                        "pan_x": { "type": "number", "description": "Horizontal focus % (0 to 100)." },
                        "pan_y": { "type": "number", "description": "Vertical focus % (0-100)." },
                        "keyword": { "type": "string", "description": "Optional: switch to a different video." }
                    }
                }
            },
            {
                "name": "close_media",
                "description": "Remove the current media from the user's screen.",
                "parameters": { "type": "object", "properties": {} }
            }
        ]
    }
]

async def bot(args: RunnerArguments):
    print(f"\n{'='*65}")
    print(f"🤖 QUADRIGA BOT (FULL PRODUCTION ATOMIC ARCHITECTURE)")
    print(f"{'='*65}\n")

    transport = None
    greeting_triggered = False
    greeting_lock = asyncio.Lock() # DOUBLE-SPEAK LOCK

    # Initialize VAD with specific Quadriga parameters
    # Performance Fix: stop_secs lowered to 0.8s to close turns faster
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            start_secs=0.15, 
            stop_secs=0.8, 
            confidence=0.55, 
            min_volume=0.45
        )
    )

    # Robust Transport Selection [Daily or SmallWebRTC]
    if isinstance(args, DailyRunnerArguments):
        logger.info(f"📹 MODE: DAILY (CLOUD) -> Joining {args.room_url}")
        transport = DailyTransport(
            room_url=args.room_url,
            token=args.token,
            bot_name="Quadriga",
            params=DailyParams(
                audio_out_enabled=True,
                audio_in_enabled=True,
                vad_enabled=True,            
                vad_audio_passthrough=True,
                camera_out_enabled=False
            )
        )
    elif isinstance(args, SmallWebRTCRunnerArguments):
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
        logger.info(f"💻 MODE: SMALLWEBRTC (LOCAL)")
        transport = SmallWebRTCTransport(
            webrtc_connection=args.webrtc_connection,
            params=DailyParams(
                audio_out_enabled=True, 
                audio_in_enabled=True, 
                vad_enabled=True, 
                vad_audio_passthrough=True
            )
        )

    # Gemini Live Setup with verified Native Audio model
    llm = GeminiLiveLLMService(
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        model="models/gemini-2.5-flash-native-audio-preview-12-2025",
        voice_id="Charon", 
        transcribe_user_audio=True,
        system_instruction=SYSTEM_PROMPT + VISUAL_INSTRUCTIONS,
        tools=tools,
        params=InputParams(
            generation_config={"response_modalities": ["AUDIO"], "temperature": 0.0}
        )
    )

    messages = []
    context = LLMContext(messages)
    
    # Modern Context Aggregator Pair (Requirement for v0.0.101)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad_analyzer)
    )
    
    rtvi = RTVIProcessor(config=RTVIConfig(config=[], enable_bot_ready_message=True))
    panic_processor = PanicButtonProcessor()
    client_notifier = ClientStateNotifier() 

    # Pipeline Assembly - The Core Loop
    pipeline = Pipeline([
        transport.input(), 
        rtvi, 
        client_notifier, 
        user_aggregator, 
        llm, 
        panic_processor, 
        transport.output(), 
        assistant_aggregator
    ])

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))

    # --- ATOMIC HANDLER: VIDEO ---
    async def update_video_state_handler(params: FunctionCallParams):
        args = params.arguments
        logger.info(f"🎬 Atomic Video Update Received: {args}")
        
        video_url = None
        if "keyword" in args:
            video_url = get_video_url(args["keyword"])

        msg = {
            "event": "update_video_state",
            "url": video_url,
            "state": {k: args.get(k) for k in ["playing", "muted", "volume", "time_abs", "time_rel", "speed", "zoom", "pan_x", "pan_y"]}
        }
        # Push frame directly to transport output to bypass any potential pipeline lag
        await transport.output().process_frame(DailyOutputTransportMessageFrame(message=msg), FrameDirection.DOWNSTREAM)
        
        # REPETITION FIX: Silent callback updates LLM without re-triggering audio
        await params.result_callback({"status": "success"})
        return "Applied."

    # --- ATOMIC HANDLER: IMAGE ---
    async def show_image_handler(params: FunctionCallParams):
        args = params.arguments
        keyword = args.get("keyword", "").lower()
        image_url = get_image_url(keyword)
        
        logger.info(f"🎨 Image Transaction: Keyword={keyword}, Zoom={args.get('zoom')}, Pan=({args.get('pan_x')},{args.get('pan_y')})")
        
        msg = {
            "event": "show_image",
            "url": image_url,
            "zoom": args.get("zoom", 1.0),
            "pan_x": args.get("pan_x", 50),
            "pan_y": args.get("pan_y", 50)
        }
        await transport.output().process_frame(DailyOutputTransportMessageFrame(message=msg), FrameDirection.DOWNSTREAM)
        
        # REPETITION FIX: Silent callback
        await params.result_callback({"status": "success"})
        return f"Displaying {keyword} image."

    # --- ATOMIC HANDLER: CLOSE ---
    async def close_media_handler(params: FunctionCallParams):
        logger.info("❌ Tool Triggered: close_media - Clearing Client Screen")
        msg = {"event": "close_image"}
        await transport.output().process_frame(DailyOutputTransportMessageFrame(message=msg), FrameDirection.DOWNSTREAM)
        await params.result_callback({"status": "success"})
        return "Media display cleared."

    # Register tools with cancel_on_interruption=False to prevent 10s timeout errors
    llm.register_function("update_video_state", update_video_state_handler, cancel_on_interruption=False)
    llm.register_function("show_image", show_image_handler, cancel_on_interruption=False)
    llm.register_function("close_media", close_media_handler, cancel_on_interruption=False)

    # --- EVENT: APP MESSAGE (Includes User Shim) ---
    @transport.event_handler("on_app_message")
    async def on_app_message(transport, message, sender):
        logger.debug(f"Received App Message from Client: {message}")
        
        if message.get("event") == "video_ended":
            logger.info("🎬 Video finished. Injecting User Shim to trigger bot dialogue.")
            await task.queue_frames([
                UserStartedSpeakingFrame(), # Mechanically wake turn management
                LLMMessagesAppendFrame(
                    messages=[{
                        "role": "user", 
                        "content": "[System Notification]: The video is done. Announce completion naturally and ask for next steps."
                    }],
                    run_llm=True 
                ),
                UserStoppedSpeakingFrame()
            ])
            
        elif message.get("event") == "video_error":
            logger.error("❌ Video element error reported by browser client.")
            await task.queue_frames([
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "The video player encountered an error. Please apologize and offer to try again."}],
                    run_llm=True
                )
            ])

    # --- EVENT: GREETING LOGIC (WITH ATOMIC LOCK) ---
    async def trigger_greeting():
        nonlocal greeting_triggered
        async with greeting_lock: # FORCES SEQUENTIAL ACCESS
            if greeting_triggered:
                return
            greeting_triggered = True
            logger.info("👋 Executing Start-of-Call Greeting Trigger.")
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

    # --- EVENT: PARTICIPANT MANAGEMENT ---
    if isinstance(args, DailyRunnerArguments):
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.info(f"Human participant detected! Participant ID: {participant['id']}")
            # Start transcription capture for the human participant
            await transport.capture_participant_transcription(participant["id"])
            await trigger_greeting()

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logger.info(f"Participant left the room ({reason}). Terminating bot session.")
            os._exit(0)

    # --- START THE RUNNER ---
    runner = PipelineRunner()
    try:
        await runner.run(task)
    except Exception as e:
        logger.exception(f"Pipeline Runner encountered a fatal error: {e}")

# -------------------------------------------------------------------------
# UTILITY: DAILY TOKEN GENERATOR (RESTORED)
# -------------------------------------------------------------------------
def get_daily_token(room_url, api_key):
    if not api_key:
        logger.error("❌ Cannot generate meeting token: DAILY_API_KEY is missing.")
        return None
    try:
        room_name = room_url.strip('/').split('/')[-1]
        url = "https://api.daily.co/v1/meeting-tokens"
        headers = {
            "Authorization": f"Bearer {api_key}", 
            "Content-Type": "application/json"
        }
        # Set token to expire in 1 hour
        expiration = int(time.time() + 3600) 
        data = {
            "properties": {
                "room_name": room_name, 
                "is_owner": True, 
                "exp": expiration 
            }
        }
        
        logger.info(f"Requesting Daily token for room: {room_name}")
        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read())
            token = res_data.get("token")
            if token:
                logger.info("Successfully acquired Daily token.")
                return token
            else:
                logger.error("Token missing from Daily API response.")
                return None
    except Exception as e:
        logger.error(f"❌ Daily API Token error: {e}")
        return None

# -------------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    ROOM_URL = "https://resqhealth.daily.co/quadriga-agent-test"
    DAILY_KEY = os.getenv("DAILY_API_KEY", "")
    
    # Generate session token before starting the bot task
    meeting_token = get_daily_token(ROOM_URL, DAILY_KEY)
    
    if meeting_token:
        logger.info("🚀 INITIALIZING QUADRIGA SESSION...")
        args = DailyRunnerArguments(room_url=ROOM_URL, token=meeting_token)
        try:
            asyncio.run(bot(args))
        except KeyboardInterrupt:
            logger.warning("Bot process manually interrupted by user.")
    else:
        logger.error("🛑 BOT STARTUP ABORTED: A valid Daily meeting token is required to join.")