import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import asyncio
import websockets
import numpy as np
import tensorflow_hub as hub
import csv
import scipy.io.wavfile as wav
import time

# YAMNet load
model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = model.class_map_path().numpy().decode('utf-8')
class_names = [row['display_name'] for row in csv.DictReader(open(class_map_path))]

# Settings
ALERT_THRESHOLD  = 0.7  # Alert trigger threshold
COOLDOWN_SECONDS = 3.0  # Cooldown time for consecutive alerts

last_alert_time = {}

async def handle_connection(websocket):
    print("ESP32 연결됨")
    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                continue

            # Receive and save audio data
            audio_data = np.frombuffer(message, dtype=np.int16)
            # wav.write("debug_listen.wav", 16000, audio_data)

            # Audio normalization
            audio_float = audio_data.astype(np.float32) / 32768.0
            max_amp = np.max(np.abs(audio_float))
            if max_amp > 0:
                audio_float = audio_float / max_amp

            # YAMNet inference
            scores, _, _ = model(audio_float)
            mean_scores = np.mean(scores, axis=0)

            # Print top 3 results
            top3 = np.argsort(mean_scores)[::-1][:3]
            print("\n[소리 분석]")
            for i in top3:
                print(f"  {class_names[i]}: {mean_scores[i] * 100:.1f}%")

            top_class = class_names[top3[0]]
            top_score = mean_scores[top3[0]]
            current_time = time.time()

            # Send alert if threshold is exceeded (with cooldown)
            if top_score >= ALERT_THRESHOLD:
                last_time = last_alert_time.get(top_class, 0)
                if current_time - last_time > COOLDOWN_SECONDS:
                    await websocket.send(f"ALERT:{top_class}")
                    print(f"[알림] {top_class} ({top_score*100:.1f}%)")
                    last_alert_time[top_class] = current_time
                else:
                    print(f"[쿨타임] {top_class} 알림 생략")
                    await websocket.send(f"RESULT:{top_class}")
            else:
                await websocket.send(f"RESULT:{top_class}")

    except Exception as e:
        print(f"에러: {e}")

async def main():
    async with websockets.serve(
        handle_connection,
        "0.0.0.0",
        8765,
        ping_interval=None,
        ping_timeout=None
    ):
        print("서버 시작 (port 8765)")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())