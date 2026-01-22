# bot.py
import os
import sys
import asyncio
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIConfig
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, InputParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.runner.types import DailyRunnerArguments, SmallWebRTCRunnerArguments, RunnerArguments

from prompts import SYSTEM_PROMPT, GREETING_TEXT

logger.remove()
logger.add(sys.stderr, level="DEBUG")

async def bot(args: RunnerArguments):
    print(f"\n{'='*40}")
    print(f"🚀 QUADRIGA BOT (Dual Mode)")
    print(f"{'='*40}\n")

    transport = None
    client_connected = False
    RECONNECT_GRACE_PERIOD = 30  # seconds

    # 1. Detect Transport Type
    if isinstance(args, DailyRunnerArguments):
        from pipecat.transports.daily.transport import DailyTransport, DailyParams

        logger.info(f"🎯 Mode: Daily (Cloud/URL)")
        logger.info(f"🎯 Room URL: {args.room_url}")
        
        transport = DailyTransport(
            room_url=args.room_url,
            token=args.token,
            bot_name="Quadriga",
            params=DailyParams(
                audio_out_enabled=True,
                transcription_enabled=False,
                audio_in_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer()
            )
        )

    elif isinstance(args, SmallWebRTCRunnerArguments):
        from pipecat.transports.base_transport import TransportParams
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        logger.info("🎯 Mode: SmallWebRTC (Local)")
        
        transport = SmallWebRTCTransport(
            webrtc_connection=args.webrtc_connection,
            params=TransportParams(
                audio_out_enabled=True,
                audio_in_enabled=True,
                vad_analyzer=SileroVADAnalyzer()
            )
        )

    else:
        logger.error(f"Unsupported runner arguments type: {type(args)}")
        return

    # LLM Setup
    llm = GeminiLiveLLMService(
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        model="models/gemini-2.5-flash-native-audio-preview-12-2025",
        voice_id="Charon", 
        transcribe_user_audio=True,
        system_instruction=SYSTEM_PROMPT, 
        params=InputParams(
            generation_config={
                "max_output_tokens": 180,
                "temperature": 0.0,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.4,
                "response_modalities": ["AUDIO"]
            }
        )
    )

    # Context setup for conversation history
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    # RTVI Processor
    rtvi = RTVIProcessor(config=RTVIConfig(
        config=[], 
        enable_bot_ready_message=True 
    ))

    pipeline = Pipeline([
        transport.input(),
        rtvi,
        user_aggregator,
        llm,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True),
    )

    # --- Event Handlers ---

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Client ready - sending bot ready")
        await rtvi.set_bot_ready()
        
        messages.append({
            "role": "system", 
            "content": f"Say exactly this greeting: \"{GREETING_TEXT}\""
        })
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal client_connected
        client_connected = True
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        nonlocal client_connected
        client_connected = False
        logger.info(f"Client disconnected. Waiting {RECONNECT_GRACE_PERIOD}s for reconnection...")
        
        await asyncio.sleep(RECONNECT_GRACE_PERIOD)
        
        if not client_connected:
            logger.info("No reconnection detected. Terminating session.")
            await task.cancel()
        else:
            logger.info("Client reconnected. Session continues.")

    runner = PipelineRunner(handle_sigint=False)
    
    try:
        await runner.run(task)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Pipeline task cancelled. Exiting cleanly.")

if __name__ == "__main__":
    from pipecat.runner.run import main
    main()