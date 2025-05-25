import asyncio
import websockets
import pyaudio
import json

WS_URI = "wss://probable-space-eureka-wr5gxpx99rwv39pv5-8001.app.github.dev/ws/stream"  # Update if hosted differently
CHUNK = 1024
RATE = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 5

async def microphone_chat():
    async with websockets.connect(WS_URI) as ws:
        print("🎤 Recording...")

        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        # Record and send audio
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            await ws.send(data)

        print("✅ Sent audio, now sending done flag")
        await ws.send(json.dumps({"type": "text_message", "message": "done"}))

        # Listen for response
        while True:
            try:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    print(f"[AUDIO] 🔊 Received TTS audio ({len(msg)} bytes)")
                else:
                    data = json.loads(msg)
                    if data["type"] == "transcript":
                        print("[STT]", data["text"])
                    elif data["type"] == "llm_response":
                        print("[LLM]", data["text"])
            except websockets.exceptions.ConnectionClosed:
                print("👋 Connection closed")
                break

        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    asyncio.run(microphone_chat())
