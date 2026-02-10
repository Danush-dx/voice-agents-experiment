"""
Simplified Voice Client for VoiceStreamAI.
Connects to the local agent, waits for user trigger, and handles the call.
Feature: Mutes microphone ONLY during the opening greeting.

Usage:
    pip install sounddevice numpy websockets
    python voice_client.py
"""
import asyncio
import json
import base64
import argparse
import sys
import time
import audioop
import queue

# Check dependencies
try:
    import websockets
    import numpy as np
    import sounddevice as sd
except ImportError as e:
    print("❌ Missing dependencies!")
    print(f"Error: {e}")
    print("\nPlease install required packages:")
    print("  pip install websockets numpy sounddevice")
    sys.exit(1)

# Audio Configuration
SAMPLE_RATE_MIC = 8000   # Changed to 8000 to match Network
SAMPLE_RATE_NET = 8000   # Network (Vobiz) rate
CHANNELS = 1
BLOCK_SIZE = 512         # Reduced block size for 8k

# Queues
mic_queue = queue.Queue()
spk_queue = queue.Queue()

class VoiceClient:
    def __init__(self, url):
        self.url = url
        self.running = True
        self.ws = None
        # self.resample_state = None # No longer needed
        
        # State Flags
        self.greeting_active = True  # Start muted
        self.has_received_audio = False # To know when greeting starts
        self.agent_speaking = False  # Echo mute: True while agent audio is playing

    def audio_callback(self, indata, outdata, frames, time, status):
        """Callback for sounddevice stream."""
        if status:
            print(f"⚠️ Audio status: {status}", file=sys.stderr)

        # --- SPEAKER OUTPUT (process first to set agent_speaking flag) ---
        try:
            data_to_play = bytearray()
            needed_bytes = frames * 2 # 16-bit target

            while len(data_to_play) < needed_bytes:
                try:
                    chunk = spk_queue.get_nowait()
                    data_to_play.extend(chunk)
                except queue.Empty:
                    break

            if len(data_to_play) > 0:
                self.agent_speaking = True
                if len(data_to_play) >= needed_bytes:
                    chunk_16 = np.frombuffer(data_to_play[:needed_bytes], dtype='int16')
                    outdata[:len(chunk_16), 0] = chunk_16 / 32768.0
                else:
                    chunk_16 = np.frombuffer(data_to_play, dtype='int16')
                    outdata[:len(chunk_16), 0] = chunk_16 / 32768.0
                    outdata[len(chunk_16):, 0] = 0
            else:
                self.agent_speaking = False
                outdata.fill(0)

        except Exception as e:
            print(f"Audio callback error: {e}", file=sys.stderr)
            self.agent_speaking = False
            outdata.fill(0)

        # --- MICROPHONE INPUT ---
        if self.greeting_active and self.has_received_audio and not self.agent_speaking:
            self.greeting_active = False
            print("\n🎙️  Listening... (You can speak now)", file=sys.stderr)

        if self.greeting_active or self.agent_speaking:
            indata.fill(0) # Send silence (mute mic during agent playback)

        mic_queue.put(indata.copy())

    async def send_audio_loop(self):
        """Consume mic queue, convert, and send to WebSocket."""
        while self.running:
            if mic_queue.empty():
                await asyncio.sleep(0.01)
                continue

            try:
                indata = mic_queue.get_nowait()
                
                # --- NOISE GATE ---
                rms = np.sqrt(np.mean(indata**2))
                if rms < 0.02: 
                     indata.fill(0)

                # Convert to bytes (16-bit PCM)
                pcm_bytes = (indata * 32767).astype(np.int16).tobytes()
                
                # No Resampling needed (8k -> 8k)
                
                # Convert to µ-law
                mulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)
                
                payload = base64.b64encode(mulaw_bytes).decode('utf-8')
                
                msg = {
                    "event": "media",
                    "media": {
                        "payload": payload,
                        "track": "inbound",
                        "chunk": "1", 
                        "timestamp": str(int(time.time() * 1000))
                    }
                }
                
                if self.ws:
                    await self.ws.send(json.dumps(msg))
                    
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed.", file=sys.stderr)
                break
            except Exception as e:
                print(f"Error sending audio: {e}", file=sys.stderr)
                continue

    async def receive_audio_loop(self):
        """Receive WebSocket messages and play audio."""
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    event = data.get("event")
                    
                    if event == "playAudio":
                        media = data.get("media", {})
                        payload = media.get("payload")
                        if payload:
                            self.has_received_audio = True
                            
                            mulaw_bytes = base64.b64decode(payload)
                            pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
                            
                            # No Resampling needed (8k -> 8k)
                            spk_queue.put(pcm_8k)
                            
                    elif event == "mark":
                        pass
                    elif event == "clearAudio":
                        with spk_queue.mutex:
                            spk_queue.queue.clear()
                            
                except Exception as e:
                    print(f"Error receiving: {e}", file=sys.stderr)
                    
        except websockets.exceptions.ConnectionClosed:
            print("Server disconnected.", file=sys.stderr)
            self.running = False

    async def run(self):
        print(f"Connecting to {self.url}...")
        async with websockets.connect(self.url) as ws:
            self.ws = ws
            print("✅ Connected to Server.")
            
            # --- MANUAL START TRIGGER ---
            # Using run_in_executor to avoid blocking asyncio loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, input, "\n👉 Press ENTER to start the session...\n")
            
            # Handshake
            print("📞 Session Started.")
            print("🔊 Agent Greeting (Mic Muted)...")
            
            await ws.send(json.dumps({"event": "connected"}))
            await ws.send(json.dumps({
                "event": "start",
                "streamSid": "voice_client",
                "callSid": "voice_client_sid",
                "start": {
                    "mediaFormat": {
                        "encoding": "audio/x-mulaw",
                        "sampleRate": 8000,
                        "channels": 1
                    }
                }
            }))
            
            # Start Audio Device
            with sd.Stream(
                samplerate=SAMPLE_RATE_MIC,
                channels=CHANNELS,
                dtype='float32',
                blocksize=BLOCK_SIZE,
                callback=self.audio_callback
            ):
                await asyncio.gather(
                    self.send_audio_loop(),
                    self.receive_audio_loop()
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8080/api/telephony/stream")
    args = parser.parse_args()

    client = VoiceClient(args.url)
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
