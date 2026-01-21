import os
import sys
import asyncio
import time
import aiohttp
import argparse
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

from prompts import SYSTEM_PROMPT, GREETING_TEXT

logger.remove()
logger.add(sys.stderr, level="DEBUG")

# --- ROOM CONFIGURATION ---
async def configure_daily_room(args):
    # Debug: Print everything we received so we can see it in the Cloud Logs
    logger.info(f"🔍 DEBUG: Sys Args: {sys.argv}")
    logger.info(f"🔍 DEBUG: Daily Env Var: {os.getenv('DAILY_ROOM_URL')}")

    # 1. Check Command Line Argument (Standard Cloud Method)
    if args.url:
        logger.info(f"🎯 Using URL from Args: {args.url}")
        return args.url

    # 2. Check Environment Variable (Fallback Cloud Method)
    if os.getenv("DAILY_ROOM_URL"):
        logger.info(f"🎯 Using URL from Env Var: {os.getenv('DAILY_ROOM_URL')}")
        return os.getenv("DAILY_ROOM_URL")

    # 3. Local Dynamic Generation (Fallback for Local Testing)
    logger.info("⚠️ No Cloud URL found. Falling back to Local Dynamic Room.")
    return await create_dynamic_room()

async def create_dynamic_room():
    api_key = os.getenv("DAILY_API_KEY")
    if not api_key:
        raise ValueError("Error: DAILY_API_KEY is missing. Cannot create room.")

    logger.info("✨ Creating a new dynamic room for this session...")
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # Create a room that expires in 1 hour
        data = {
            "properties": {
                "exp": int(time.time() + 3600),
                "eject_at_room_exp": True
            }
        }
        
        async with session.post("https://api.daily.co/v1/rooms", headers=headers, json=data) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise ValueError(f"Failed to create Daily room: {text}")
            
            room_data = await resp.json()
            new_url = room_data["url"]
            logger.info(f"Successfully created dynamic room: {new_url}")
            return new_url

# ----------------------------------------------------

async def main(args):
    # Banner to verify code version in logs
    print(f"\n{'='*40}")
    print(f"🚀 QUADRIGA BOT v2.3 (Token Support)")
    print(f"{'='*40}\n")

    try:
        daily_url = await configure_daily_room(args)
    except ValueError as e:
        logger.error(str(e))
        return

    # UPDATED: Pass args.token to the transport. 
    # This is critical for Cloud Sandbox which sends a token for authentication.
    transport = DailyTransport(
        room_url=daily_url,
        token=args.token,  # <--- FIXED: Use the token passed by the runner
        bot_name="Quadriga",
        params=DailyParams(
            audio_out_enabled=True,
            transcription_enabled=False,
            audio_in_enabled=True
        )
    )

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

    # RTVI: Critical for Pipecat Cloud UI to recognize the bot
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
        logger.info(f"Client connected to {daily_url}")
        
        # 1. Prepare Bot Ready Signal
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

        # 2. Prepare Greeting
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
    parser = argparse.ArgumentParser(description="Quadriga Agent")
    parser.add_argument("-u", "--url", type=str, help="Room URL (Passed by Pipecat Cloud)")
    # UPDATED: Added token argument to prevent crash in Cloud Runner
    parser.add_argument("-t", "--token", type=str, help="Daily Room Token (Passed by Pipecat Cloud)")
    args = parser.parse_args()

    asyncio.run(main(args))