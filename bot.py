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
    EndFrame
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIConfig

# FIXED IMPORTS FOR v0.0.101
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

# Restore full logging configuration
logger.remove()
logger.add(sys.stderr, level="DEBUG")

# -------------------------------------------------------------------------
# 1. CLIENT STATE NOTIFIER (Audio Ducking Logic)
# -------------------------------------------------------------------------
class ClientStateNotifier(FrameProcessor):
    def __init__(self):
        super().__init__()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            msg = DailyOutputTransportMessageFrame(message={
                "event": "user_speaking_status", 
                "status": "speaking"
            })
            await self.push_frame(msg, direction)
            
        elif isinstance(frame, UserStoppedSpeakingFrame):
            msg = DailyOutputTransportMessageFrame(message={
                "event": "user_speaking_status", 
                "status": "stopped"
            })
            await self.push_frame(msg, direction)

        await self.push_frame(frame, direction)

# -------------------------------------------------------------------------
# 2. PANIC BUTTON PROCESSOR (Downgraded Fallback)
# -------------------------------------------------------------------------
# Reduced to ONLY provide low-latency emergency stops to prevent race conditions.
class PanicButtonProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        
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
                    logger.info(f"🚨 Panic Button Triggered: Detected '{command}'")
                    msg_payload = {"event": "video_control", "command": command, "source": "panic_button"}
                    await self.push_frame(DailyOutputTransportMessageFrame(message=msg_payload), direction)
                    self._triggered_intents.add(command)

        await self.push_frame(frame, direction)

# -------------------------------------------------------------------------
# ATOMIC TOOL DEFINITIONS
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
                        "keyword": { "type": "string", "description": "Image keyword (e.g. 'birthday')." }
                    },
                    "required": ["keyword"]
                }
            },
            {
                "name": "update_video_state",
                "description": "Atomic control tool. Calculate the desired state and send in one transaction.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "playing": { "type": "boolean", "description": "True to play, False to pause." },
                        "muted": { "type": "boolean", "description": "True to mute audio, False to unmute." },
                        "time_abs": { "type": "number", "description": "Jump to absolute second (0 to restart)." },
                        "time_rel": { "type": "number", "description": "Seek +/- seconds relative to current time." },
                        "speed": { "type": "number", "description": "Playback rate (0.5 to 2.0)." },
                        "keyword": { "type": "string", "description": "Optional keyword to switch media source." }
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
    print(f"\n{'='*40}")
    print(f"🤖 QUADRIGA BOT (Full Atomic Implementation)")
    print(f"{'='*40}\n")

    transport = None
    greeting_triggered = False

    # VAD settings
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(start_secs=0.2, stop_secs=1.5, confidence=0.7, min_volume=0.6)
    )

    if isinstance(args, DailyRunnerArguments):
        logger.info(f"📹 Mode: Daily (Cloud/URL) -> {args.room_url}")
        transport = DailyTransport(
            room_url=args.room_url,
            token=args.token,
            bot_name="Quadriga",
            params=DailyParams(
                audio_out_enabled=True,
                audio_in_enabled=True,
                vad_enabled=True,            
                vad_audio_passthrough=True   
            )
        )
    elif isinstance(args, SmallWebRTCRunnerArguments):
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
        logger.info(f"💻 Mode: SmallWebRTC (Local)")
        transport = SmallWebRTCTransport(
            webrtc_connection=args.webrtc_connection,
            params=DailyParams(
                audio_out_enabled=True, 
                audio_in_enabled=True, 
                vad_enabled=True, 
                vad_audio_passthrough=True
            )
        )

    # Gemini Live Setup using proven model identifier
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

    # Universal Aggregator setup
    messages = []
    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad_analyzer)
    )
    
    rtvi = RTVIProcessor(config=RTVIConfig(config=[], enable_bot_ready_message=True))
    panic_processor = PanicButtonProcessor()
    state_notifier = ClientStateNotifier() 

    pipeline = Pipeline([
        transport.input(), 
        rtvi, 
        state_notifier, 
        user_aggregator, 
        llm, 
        panic_processor, 
        transport.output(), 
        assistant_aggregator
    ])

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))

    # --- ATOMIC HANDLERS ---
    async def update_video_state_handler(params: FunctionCallParams):
        args = params.arguments
        logger.info(f"🎬 Atomic Transaction Received: {args}")
        
        video_url = None
        if "keyword" in args:
            video_url = get_video_url(args["keyword"])

        msg = {
            "event": "update_video_state",
            "url": video_url,
            "state": {
                "playing": args.get("playing"),
                "muted": args.get("muted"),
                "time_abs": args.get("time_abs"),
                "time_rel": args.get("time_rel"),
                "speed": args.get("speed")
            }
        }
        await transport.output().process_frame(DailyOutputTransportMessageFrame(message=msg), FrameDirection.DOWNSTREAM)
        # FIX: Explicitly call result callback to prevent 10s tool timeouts
        await params.result_callback({"status": "success", "applied_state": args})
        return "Atomic state updated."

    async def show_image_handler(params: FunctionCallParams):
        keyword = params.arguments.get("keyword", "").lower()
        image_url = get_image_url(keyword)
        logger.info(f"🎨 Tool Triggered: show_image ({keyword})")
        msg = {"event": "show_image", "url": image_url}
        await transport.output().process_frame(DailyOutputTransportMessageFrame(message=msg), FrameDirection.DOWNSTREAM)
        await params.result_callback({"status": "success", "image_url": image_url})
        return f"Displaying image for {keyword}."

    async def close_media_handler(params: FunctionCallParams):
        logger.info("❌ Tool Triggered: close_media")
        msg = {"event": "close_image"}
        await transport.output().process_frame(DailyOutputTransportMessageFrame(message=msg), FrameDirection.DOWNSTREAM)
        await params.result_callback({"status": "success"})
        return "Media closed."

    # CRITICAL: cancel_on_interruption=False ensures tool completion
    llm.register_function("update_video_state", update_video_state_handler, cancel_on_interruption=False)
    llm.register_function("show_image", show_image_handler, cancel_on_interruption=False)
    llm.register_function("close_media", close_media_handler, cancel_on_interruption=False)

    # --- THE USER SHIM (Guaranteed turn-taking when video ends) ---
    @transport.event_handler("on_app_message")
    async def on_app_message(transport, message, sender):
        if message.get("event") == "video_ended":
            logger.info("🎬 Video finished. Injecting User Shim.")
            await task.queue_frames([
                UserStartedSpeakingFrame(), # Mechanical wake-up for aggregators
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "[System]: The video is done. Announce this and ask what to do next."}],
                    run_llm=True 
                ),
                UserStoppedSpeakingFrame()
            ])
            
        elif message.get("event") == "video_error":
            logger.error("❌ Video failed.")
            await task.queue_frames([
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "The video failed to load. Please apologize."}],
                    run_llm=True
                )
            ])

    # Greeting Trigger Logic
    async def trigger_greeting():
        nonlocal greeting_triggered
        if greeting_triggered: return
        greeting_triggered = True
        logger.info("👋 Executing Greeting Trigger.")
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

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logger.info("User left. Terminating session.")
            os._exit(0)

    # Final Pipeline Runner
    runner = PipelineRunner()
    await runner.run(task)

# Utility Logic Restored
def get_daily_token(room_url, api_key):
    if not api_key: return None
    try:
        room_name = room_url.strip('/').split('/')[-1]
        url = "https://api.daily.co/v1/meeting-tokens"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = { "properties": { "room_name": room_name, "is_owner": True, "exp": int(time.time() + 3600) } }
        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())["token"]
    except Exception as e:
        logger.error(f"❌ Token error: {e}")
        return None

if __name__ == "__main__":
    ROOM_URL = "https://resqhealth.daily.co/quadriga-agent-test"
    API_KEY = os.getenv("DAILY_API_KEY", "")
    meeting_token = get_daily_token(ROOM_URL, API_KEY)
    
    if meeting_token:
        args = DailyRunnerArguments(room_url=ROOM_URL, token=meeting_token)
        asyncio.run(bot(args))
    else:
        logger.error("🛑 Failed to acquire session token.")