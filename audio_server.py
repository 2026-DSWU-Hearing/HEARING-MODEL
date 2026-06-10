import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import asyncio
import websockets
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import csv
import time

from category_map import get_category_info

# YAMNet load
model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = model.class_map_path().numpy().decode('utf-8')
class_names = [row['display_name'] for row in csv.DictReader(open(class_map_path))]

ALERT_THRESHOLD = 0.7
COOLDOWN_SECONDS = 3.0
last_alert_time = {}

async def handle_connection(websocket):
    print("ESP32 연결됨")
    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                continue

            audio_data = np.frombuffer(message, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            max_amp = np.max(np.abs(audio_float))
            if max_amp > 0:
                audio_float = audio_float / max_amp

            scores, _, _ = model(audio_float)
            mean_scores = tf.reduce_mean(scores, axis=0).numpy()

            # top3 = np.argsort(mean_scores)[::-1][:3]
            # print("\nYAMNet top3")
            # for i in top3:
            #     print(f"  {class_names[i]}: {mean_scores[i]*100:.1f}%")

            top_indices = mean_scores.argsort()[-20:][::-1]
            block_results = {}
            for idx in top_indices:
                sound_name = class_names[idx]
                score = float(mean_scores[idx])
                if score < 0.1:
                    continue
                category, block = get_category_info(sound_name)
                if category == "기타" or block == "기타":
                    continue
                key = (category, block)
                if key not in block_results or score > block_results[key]["score"]:
                    block_results[key] = {
                        "category": category,
                        "block": block,
                        "score": score
                    }

            if not block_results:
                continue

            top = sorted(block_results.values(), key=lambda x: x["score"], reverse=True)[0]
            sound_name = top["block"]
            sound_category = top["category"]
            top_score = top["score"]
            current_time = time.time()

            print(f"\n[소리 분석] {sound_category} - {sound_name}: {top_score*100:.1f}%")

            if top_score >= ALERT_THRESHOLD:
                last_time = last_alert_time.get(sound_name, 0)
                if current_time - last_time > COOLDOWN_SECONDS:
                    await websocket.send(f"ALERT:{sound_name}")
                    print(f"[알림] {sound_name} ({top_score*100:.1f}%)")
                    last_alert_time[sound_name] = current_time

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
        print("AI 서버 시작 (port 8765)")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())