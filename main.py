import logging
import os
import json
import tempfile
import asyncio
from typing import Dict

import uvicorn
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState

# --- Configuration ---
FASTER_WHISPER_URL = "http://34.93.142.196:8000/v1/transcriptions"
QWEN_URL = "http://34.93.142.196:8005/generate"
KOKOROTTS_URL = "http://34.93.142.196:8880/v1/audio/speech"
BEARER_TOKEN = "dummy_api_key"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI Setup ---
app = FastAPI(title="Real-Time Voice AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session Management ---
active_sessions = {}

class VoiceSession:
    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.is_active = True
        
        # Async Queues
        self.audio_queue = asyncio.Queue()
        self.text_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        
        # Tasks
        self.tasks = []
    
    async def start(self):
        # Start the worker tasks
        self.tasks = [
            asyncio.create_task(self._stt_worker()),
            asyncio.create_task(self._llm_worker()),
            asyncio.create_task(self._tts_worker())
        ]
        
    async def stop(self):
        self.is_active = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete cancellation
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks = []
    
    async def add_audio_chunk(self, audio_chunk: bytes):
        await self.audio_queue.put(audio_chunk)
    
    async def add_text_message(self, text: str):
        await self.text_queue.put(text)
    
    async def cleanup(self):
        logger.info(f"Cleaning up session {self.session_id}")
        await self.stop()
    
    async def _stt_worker(self):
        logger.info(f"STT worker started for session {self.session_id}")
        buffer = []
        silence_counter = 0
        
        try:
            while self.is_active:
                try:
                    # Use asyncio.wait_for to implement a timeout
                    chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
                    buffer.append(chunk)
                    silence_counter = 0
                except asyncio.TimeoutError:
                    silence_counter += 1
                
                if silence_counter >= 5 and buffer:
                    full_audio = b''.join(buffer)
                    transcript = await self._call_faster_whisper(full_audio)
                    
                    if transcript:
                        logger.info(f"Transcription: {transcript}")
                        await self._send_text_to_client({"type": "transcript", "text": transcript})
                        await self.text_queue.put(transcript)
                    
                    buffer.clear()
                    silence_counter = 0
        except asyncio.CancelledError:
            logger.info(f"STT worker for session {self.session_id} was cancelled")
        except Exception as e:
            logger.exception(f"Error in STT worker: {e}")
    
    async def _call_faster_whisper(self, audio_bytes: bytes) -> str:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            # Use asyncio.to_thread for blocking IO operations
            def whisper_request():
                with open(tmp_path, "rb") as f:
                    files = {"file": ("audio.wav", f, "audio/wav")}
                    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
                    response = requests.post(FASTER_WHISPER_URL, headers=headers, files=files)
                return response
            
            response = await asyncio.to_thread(whisper_request)
            
            # Clean up the temporary file
            await asyncio.to_thread(os.remove, tmp_path)
            
            if response.status_code == 200:
                return response.json().get("text", "")
            else:
                logger.error(f"Transcription API failed: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.exception(f"Error calling FasterWhisper: {e}")
            return ""
    
    async def _llm_worker(self):
        logger.info(f"LLM worker started for session {self.session_id}")
        try:
            while self.is_active:
                try:
                    user_text = await asyncio.wait_for(self.text_queue.get(), timeout=0.1)

                    logger.info(f"[LLM] Received prompt: {user_text}")

                    def llm_request():
                        headers = {"Content-Type": "application/json"}
                        payload = {
                            "prompt": user_text,
                            "max_tokens": 512,
                            "temperature": 0.7
                        }
                        return requests.post(QWEN_URL, headers=headers, json=payload)

                    res = await asyncio.to_thread(llm_request)

                    if res.status_code == 200:
                        response_data = res.json()
                        response = response_data.get("response", "Sorry, no response from Qwen.")
                    else:
                        logger.error(f"[LLM] Qwen error: {res.status_code} - {res.text}")
                        response = "Qwen API error."

                    logger.info(f"[LLM] Sending response: {response}")
                    await self._send_text_to_client({"type": "llm_response", "text": response})
                    await self.response_queue.put(response)

                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info(f"LLM worker for session {self.session_id} was cancelled")
        except Exception as e:
            logger.exception(f"LLM error: {e}")

    async def _tts_worker(self):
        logger.info(f"TTS worker started for session {self.session_id}")
        try:
            while self.is_active:
                try:
                    response_text = await asyncio.wait_for(self.response_queue.get(), timeout=0.1)

                    logger.info(f"[TTS] Converting to speech: {response_text}")

                    def tts_request():
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {BEARER_TOKEN}"
                        }
                        payload = {
                            "text": response_text,
                            "voice": "en_us_001",
                            "model": "default"
                        }
                        return requests.post(KOKOROTTS_URL, headers=headers, json=payload)

                    res = await asyncio.to_thread(tts_request)

                    if res.status_code == 200:
                        audio_bytes = res.content
                        logger.info(f"[TTS] Sending {len(audio_bytes)} bytes of audio to client.")
                        await self._send_audio_to_client(audio_bytes)
                    else:
                        logger.error(f"[TTS] KokoroTTS failed: {res.status_code} - {res.text}")

                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info(f"TTS worker for session {self.session_id} was cancelled")
        except Exception as e:
            logger.exception(f"TTS error: {e}")

    async def _send_text_to_client(self, data: Dict):
        try:
            message = json.dumps(data)
            logger.debug(f"[WebSocket] Sending text: {message}")
            await self.websocket.send_text(message)
        except Exception as e:
            logger.exception(f"WebSocket text send error: {e}")
            self.is_active = False

    async def _send_audio_to_client(self, audio_bytes: bytes):
        try:
            logger.debug(f"[WebSocket] Sending audio ({len(audio_bytes)} bytes)")
            await self.websocket.send_bytes(audio_bytes)
        except Exception as e:
            logger.exception(f"WebSocket audio send error: {e}")
            self.is_active = False

# Keep all imports and configuration exactly as you've written them...

# (... your existing imports/configuration go here...)

# Use only one WebSocket endpoint (no need to duplicate `/ws/stream`)
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    logger.info(f"Client connected: {client_id}")
    
    session = VoiceSession(client_id, websocket)
    active_sessions[client_id] = session
    
    # Start the session worker tasks
    await session.start()
    
    try:
        await session._send_text_to_client({
            "type": "connection_status",
            "status": "connected"
        })

        while websocket.application_state == WebSocketState.CONNECTED:
            try:
                message = await websocket.receive()
                if "bytes" in message:
                    await session.add_audio_chunk(message["bytes"])
                elif "text" in message:
                    try:
                        data = json.loads(message["text"])
                        if data.get("type") == "text_message":
                            await session.add_text_message(data.get("message", ""))
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON received")
            except WebSocketDisconnect:
                logger.info(f"Client disconnected: {client_id}")
                break
            except RuntimeError as re:
                logger.warning(f"Runtime WebSocket error: {re}")
                break
            except Exception as e:
                logger.exception(f"Unexpected WebSocket error: {e}")
                break

    finally:
        await session.cleanup()
        active_sessions.pop(client_id, None)

# Health check route
@app.get("/health")
def health():
    return {"status": "healthy", "active_sessions": len(active_sessions)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
