import asyncio
import json
import sys
import websockets
import io
from pydub import AudioSegment
from pydub.playback import play

# CHANGE THIS to your server's actual endpoint
WS_URI = "wss://probable-space-eureka-wr5gxpx99rwv39pv5-8000.app.github.dev/ws/stream"
CHUNK_SIZE = 17640  # ~0.2 sec of audio at 22050 Hz

async def stream_audio(path):
    async with websockets.connect(WS_URI) as ws:
        print("✅ Connected to WebSocket server")

        stop_event = asyncio.Event()

        # Output holders
        audio_count = 0
        full_audio_bytes = b""
        llm_text = ""
        transcript_text = ""

        async def receive_loop():
            nonlocal audio_count, full_audio_bytes, llm_text, transcript_text

            try:
                while not stop_event.is_set():
                    msg = await ws.recv()

                    if isinstance(msg, bytes):
                        audio_count += 1
                        print(f"[AUDIO] 🔊 Playing audio chunk #{audio_count}")
                        full_audio_bytes += msg
                        try:
                            audio = AudioSegment.from_file(io.BytesIO(msg), format="wav")
                            play(audio)
                        except Exception as e:
                            print(f"[ERROR] 🎧 Failed to play audio: {e}")

                    else:
                        try:
                            data = json.loads(msg)
                            msg_type = data.get("type")

                            if msg_type == "transcript":
                                transcript_text = data.get("text", "")
                                print(f"[STT] ✍️ Transcription: {transcript_text}")
                            elif msg_type == "llm_response":
                                llm_text = data.get("text", "")
                                print(f"[LLM] 🤖 Response: {llm_text}")
                            elif msg_type == "connection_status":
                                print(f"[Status] {data.get('status')}")
                            else:
                                print("[JSON] 📦 Received:", data)
                        except json.JSONDecodeError:
                            print("[ERROR] ❗ Invalid JSON:", msg)

            except websockets.ConnectionClosed:
                print("[CLOSED] 🚪 WebSocket connection closed.")
            finally:
                stop_event.set()

        recv_task = asyncio.create_task(receive_loop())

        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    await ws.send(chunk)
                    await asyncio.sleep(0.2)

            await ws.send(json.dumps({"type": "text_message", "message": "done"}))
            print("✅ Finished sending audio")

            await asyncio.sleep(10)  # allow response to process
        finally:
            stop_event.set()
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

            # Save LLM output
            if llm_text:
                with open("llm_output.txt", "w") as f:
                    f.write(llm_text)
                print("💾 LLM response saved to llm_output.txt")

            # Save transcription
            if transcript_text:
                with open("transcript.txt", "w") as f:
                    f.write(transcript_text)
                print("💾 Transcript saved to transcript.txt")

            # Save TTS audio
            if full_audio_bytes:
                with open("tts_output.wav", "wb") as f:
                    f.write(full_audio_bytes)
                print("💾 TTS audio saved to tts_output.wav")

            print("👋 Client shutdown cleanly.")

if __name__ == "__main__":
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
    asyncio.run(stream_audio(wav_path))
