# bot.py
import os
import sys
import asyncio
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from pipecat.frames.frames import OutputTransportMessageFrame, LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIConfig
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, InputParams
from pipecat.transports.daily.transport import DailyTransport, DailyParams
from pipecat.audio.vad.silero import SileroVADAnalyzer

from prompts import SYSTEM_PROMPT, GREETING_TEXT

logger.remove()
logger.add(sys.stderr, level="DEBUG")

async def bot(args):
    """
    Main entry point for the bot.
    The 'args' object is provided by the Pipecat runner and contains 
    the room_url and token, regardless of whether you are running locally
    or on Pipecat Cloud.
    """
    print(f"\n{'='*40}")
    print(f"🚀 QUADRIGA BOT (Dual Mode)")
    print(f"{'='*40}\n")

    logger.info(f"🎯 Room URL: {args.room_url}")
    
    # Transport setup
    # Note: We added SileroVADAnalyzer as suggested by the HelpBot. 
    # This significantly improves interruption handling.
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

    # LLM Setup - Preserving your specific configuration
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

    # RTVI: Critical for Pipecat Cloud UI
    rtvi = RTVIProcessor(config=RTVIConfig(
        config=[], 
        enable_bot_ready_message=True 
    ))

    pipeline = Pipeline([transport.input(), rtvi, llm, transport.output()])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        
        # 1. Bot Ready Signal (RTVI)
        rtvi_payload = {
            "type": "bot-ready", 
            "label": "Quadriga", 
            "id": "init", 
            "data": {
                "config": [], 
                "version": "1.0.0",
                "rtvi_client_version": "0.2.0" 
            }
        }
        await task.queue_frames([OutputTransportMessageFrame(message=rtvi_payload)])

        # 2. Initial Greeting
        greeting_content = f"""The user has joined. Say exactly this: "{GREETING_TEXT}" """
        messages = [{"role": "user", "content": greeting_content}]
        await task.queue_frames([LLMMessagesAppendFrame(messages)])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        if participant.get("info", {}).get("userName") != "Quadriga":
            logger.info(f"User left. Shutting down agent.")
            await task.cancel()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected.")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    
    try:
        await runner.run(task)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Pipeline task cancelled. Exiting cleanly.")

if __name__ == "__main__":
    # This acts as a bridge. When run locally, it sets up the environment.
    # When deployed, Pipecat Cloud calls the 'bot' function directly.
    from pipecat.runner.run import main
    main()