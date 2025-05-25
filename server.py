import asyncio
import uuid
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    logger.info(f"Client connected: {client_id}")

    stop_event = asyncio.Event()
    audio_data = bytearray()
    transcript = ""
    llm_response = ""
    tts_audio = b""

    async def stt_worker():
        nonlocal transcript
        await asyncio.sleep(1)
        transcript = "Transcribed dummy text"
        await websocket.send_json({"type": "transcript", "text": transcript})
        logger.info(f"STT worker done: {transcript}")

    async def llm_worker():
        nonlocal llm_response
        await asyncio.sleep(1)
        llm_response = f"LLM response to: {transcript}"
        await websocket.send_json({"type": "llm_response", "text": llm_response})
        logger.info(f"LLM worker done: {llm_response}")

    async def tts_worker():
        nonlocal tts_audio
        await asyncio.sleep(1)
        tts_audio = b"WAV" + b"FAKEAUDIO"  # Fake binary
        await websocket.send_bytes(tts_audio)
        logger.info("TTS worker done (fake audio sent)")

    try:
        tasks = [
            asyncio.create_task(stt_worker()),
            asyncio.create_task(llm_worker()),
            asyncio.create_task(tts_worker()),
        ]

        while not stop_event.is_set():
            try:
                msg = await websocket.receive()
            except RuntimeError as e:
                logger.warning(f"WS receive error: {e}")
                break

            if "bytes" in msg:
                audio_data.extend(msg["bytes"])
            elif "text" in msg:
                if msg["text"] == "done":
                    stop_event.set()
            else:
                logger.warning(f"Unknown message format: {msg}")

        await asyncio.gather(*tasks)

    except WebSocketDisconnect:
        logger.info("connection closed")
    finally:
        logger.info(f"Cleaning up session {client_id}")
        for task in tasks:
            task.cancel()

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False)
