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

# ✅ FIXED IMPORTS FOR v0.0.101
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import LLMAssistantContextAggregator
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator, 
    LLMUserAggregatorParams
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

logger.remove()
logger.add(sys.stderr, level="DEBUG")

# -------------------------------------------------------------------------
# 1. CLIENT STATE NOTIFIER (Audio Ducking)
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
# 2. INTENT FALLBACK PROCESSOR (The "Eavesdropper" Safety Net)
# -------------------------------------------------------------------------
class IntentFallbackProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        
        self.intent_patterns = [
            (re.compile(r"\bpausing\s*(?:the\s+)?video\b", re.IGNORECASE), "pause", None),
            (re.compile(r"\bresuming\s*(?:the\s+)?video\b", re.IGNORECASE), "play", None),
            (re.compile(r"\bmuting\s*(?:the\s+)?video\b", re.IGNORECASE), "mute", None),
            (re.compile(r"\bunmuting\s*(?:the\s+)?video\b", re.IGNORECASE), "unmute", None),
            (re.compile(r"\brestarting\s*(?:the\s+)?video\b", re.IGNORECASE), "restart", None),
            (re.compile(r"\bskipping\s*forward\b", re.IGNORECASE), "skip", None),
            (re.compile(r"\brewinding\s*(?:the\s+)?video\b", re.IGNORECASE), "rewind", None),
            (re.compile(r"\bslow(?:ing)?\s*down\b", re.IGNORECASE), "speed", 0.5),
            (re.compile(r"\bhalf\s*speed\b", re.IGNORECASE), "speed", 0.5),
            (re.compile(r"\b0?\.5x?\b", re.IGNORECASE), "speed", 0.5),
            (re.compile(r"\bquarter\s*speed\b", re.IGNORECASE), "speed", 0.25),
            (re.compile(r"\b0?\.25x?\b", re.IGNORECASE), "speed", 0.25),
            (re.compile(r"\bspeed(?:ing)?\s*up\b", re.IGNORECASE), "speed", 1.5),
            (re.compile(r"\bfast\s*forward\b", re.IGNORECASE), "speed", 1.5),
            (re.compile(r"\bdouble\s*speed\b", re.IGNORECASE), "speed", 2.0),
            (re.compile(r"\b2(?:\.0)?x\b", re.IGNORECASE), "speed", 2.0), 
            (re.compile(r"\bnormal\s*speed\b", re.IGNORECASE), "speed", 1.0),
            (re.compile(r"\b1x\s*speed\b", re.IGNORECASE), "speed", 1.0),
        ]
        self._triggered_intents = set()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseEndFrame):
            self._triggered_intents.clear()

        if isinstance(frame, TextFrame):
            text = frame.text
            for pattern, command, value in self.intent_patterns:
                if command not in self._triggered_intents and pattern.search(text):
                    logger.info(f"🕵️ Eavesdropper Safety Net: Detected '{command}' (val={value}) in text: '{text}'")
                    msg_payload = {"event": "video_control", "command": command, "source": "fallback_processor"}
                    if value is not None:
                        msg_payload["value"] = value
                    
                    await self.push_frame(DailyOutputTransportMessageFrame(message=msg_payload), direction)
                    self._triggered_intents.add(command)

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
                        "keyword": { "type": "string", "description": "Image keyword (e.g. 'birthday')." }
                    },
                    "required": ["keyword"]
                }
            },
            {
                "name": "show_video",
                "description": "Display and play a video.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": { "type": "string", "description": "Video keyword (e.g. 'skiing')." }
                    },
                    "required": ["keyword"]
                }
            },
            {
                "name": "close_image",
                "description": "Close the currently displayed image or video.",
                "parameters": { "type": "object", "properties": {} }
            },
            {
                "name": "control_video",
                "description": "Control video playback (pause, play, seek, speed).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["pause", "play", "mute", "unmute", "restart", "seek", "speed"],
                            "description": "The action to perform."
                        },
                        "value": {
                            "type": "number",
                            "description": "For 'seek': seconds (negative=back). For 'speed': playback rate (e.g. 0.25, 0.5, 1.0, 1.5, 2.0)."
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
    print(f"🤖 QUADRIGA BOT (Enhanced)")
    print(f"{'='*40}\n")

    transport = None
    greeting_triggered = False

    # --- TRANSPORT SETUP ---
    # Relaxed VAD settings to prevent cutting off complex commands
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(start_secs=0.2, stop_secs=1.5, confidence=0.7, min_volume=0.6)
    )

    if isinstance(args, DailyRunnerArguments):
        logger.info(f"📹 Mode: Daily (Cloud/URL) -> {args.room_url}")
        
        # CHANGED: No vad_analyzer passed here in v0.0.101
        transport = DailyTransport(
            room_url=args.room_url,
            token=args.token,
            bot_name="Quadriga",
            params=DailyParams(
                audio_out_enabled=True,
                transcription_enabled=False,
                audio_in_enabled=True,
                vad_enabled=True,            
                vad_audio_passthrough=True   
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
                vad_enabled=True, 
                vad_audio_passthrough=True
            )
        )

    # --- LLM SETUP ---
    combined_system_prompt = SYSTEM_PROMPT + VISUAL_INSTRUCTIONS + """
# VIDEO CONTROL RULES

1. COMPOUND COMMANDS (CRITICAL):
   - If user says "Restart and play at 2x", you must trigger BOTH.
   - You MUST verbally confirm the specifics to trigger the controls.
   - Say: "Restarting video at 2x." (This phrase triggers the actions).

2. SPEED HANDLING:
   - If user asks for specific speed (e.g. 2x, 0.5x), DO NOT say generic phrases like "Speeding up".
   - You MUST say the specific number: "Playing at 2x" or "Playing at 0.5x".
   - This exact phrasing is required to make the video player work.

TRIGGER PHRASES (Speak these exactly):
- "Pausing video"
- "Resuming video"
- "Restarting video"
- "Playing at 2x" 
- "Playing at 0.5x"
"""

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
            generation_config={"response_modalities": ["AUDIO"], "temperature": 0.0},
            safety_settings=safety_settings
        )
    )

    # --- PIPELINE SETUP ---
    messages = []
    context = LLMContext(messages)
    
    # ✅ FIXED AGGREGATORS (v0.0.101)
    # 1. User side: Uses Universal Aggregator + VAD params
    user_aggregator = LLMUserAggregator(
        context,
        params=LLMUserAggregatorParams(vad_analyzer=vad_analyzer)
    )

    # 2. Assistant side: Uses Context Aggregator (avoids the 'append' crash)
    assistant_aggregator = LLMAssistantContextAggregator(context)
    
    rtvi = RTVIProcessor(config=RTVIConfig(config=[], enable_bot_ready_message=True))
    
    fallback_processor = IntentFallbackProcessor()
    client_state_notifier = ClientStateNotifier() 

    pipeline = Pipeline([
        transport.input(), 
        rtvi, 
        client_state_notifier, 
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
            msg = {"event": "show_image", "url": image_url}
            await transport.output().process_frame(DailyOutputTransportMessageFrame(message=msg), FrameDirection.DOWNSTREAM)
        except Exception as e:
            logger.error(f"❌ Failed to send frame: {e}")
        return f"Displaying image for {keyword}."

    async def show_video_handler(params: FunctionCallParams):
        keyword = params.arguments.get("keyword", "skiing").lower()
        video_url = get_video_url(keyword)
        if not video_url:
            video_url = get_video_url("skiing")
        logger.info(f"🎬 Tool Triggered: show_video ({keyword})")
        try:
            msg = {"event": "show_video", "url": video_url}
            await transport.output().process_frame(DailyOutputTransportMessageFrame(message=msg), FrameDirection.DOWNSTREAM)
        except Exception as e:
            logger.error(f"❌ Failed to send frame: {e}")
        return f"Video displayed on screen."

    async def close_image_handler(params: FunctionCallParams):
        logger.info("❌ Tool Triggered: close_image")
        try:
            msg = {"event": "close_image"}
            await transport.output().process_frame(DailyOutputTransportMessageFrame(message=msg), FrameDirection.DOWNSTREAM)
        except Exception as e:
            logger.error(f"❌ Failed to send frame: {e}")
        return "Media closed."

    async def control_video_handler(params: FunctionCallParams):
        action = params.arguments.get("action")
        value = params.arguments.get("value")
        
        logger.info(f"🎬 Tool Triggered: control_video -> {action} ({value})")
        
        msg = {"event": "video_control", "command": action, "source": "tool_call"}
        if value is not None:
            msg["value"] = value

        try:
            await transport.output().process_frame(DailyOutputTransportMessageFrame(message=msg), FrameDirection.DOWNSTREAM)
        except Exception as e:
            logger.error(f"❌ Failed to send video control frame: {e}")
        
        if action == "seek":
            direction = "back" if value < 0 else "forward"
            return f"Skipping {direction} {abs(value)} seconds."
        elif action == "speed":
            return f"Setting playback speed to {value}x."
        
        return f"Video {action} command sent."

    llm.register_function("show_image", show_image_handler, cancel_on_interruption=False)
    llm.register_function("show_video", show_video_handler, cancel_on_interruption=False)
    llm.register_function("close_image", close_image_handler, cancel_on_interruption=False)
    llm.register_function("control_video", control_video_handler, cancel_on_interruption=False)

    # --- APP MESSAGE HANDLER (Video End) ---
    @transport.event_handler("on_app_message")
    async def on_app_message(transport, message, sender):
        if message.get("event") == "video_ended":
            logger.info("🎬 Video finished. Forcing conversation.")
            await task.queue_frames([
                LLMMessagesAppendFrame(
                    messages=[{
                        "role": "user", 
                        "content": "The video just finished playing on the screen. Please announce that it is done and ask me if I want to replay it or move on."
                    }],
                    run_llm=True 
                )
            ])
            
        elif message.get("event") == "video_error":
            logger.error("❌ Video failed.")
            await task.queue_frames([
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "The video failed to load. Please apologize."}],
                    run_llm=True
                )
            ])

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

        # ✅ NUCLEAR EXIT HANDLER: Kills process immediately
        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logger.info(f"User left: {participant}. Terminating.")
            os._exit(0)

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