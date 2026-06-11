import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
import logging
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from app.data.category_map import get_category_info

logger = logging.getLogger(__name__)

_YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
_MIN_SCORE = 0.1
_TOP_K = 20


class AudioClassifier:
    def __init__(self):
        self._model = hub.load(_YAMNET_URL)
        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        self._class_names = [
            row["display_name"] for row in csv.DictReader(open(class_map_path))
        ]

    def classify(self, audio_bytes: bytes) -> dict | None:
        """
        Returns the top-scoring non-기타 block result, or None if nothing qualifies.
        Result shape: {"category": str, "block": str, "score": float}
        """
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio = audio / max_amp

        scores, _, _ = self._model(audio)
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()

        top_indices = mean_scores.argsort()[-_TOP_K:][::-1]
        block_results: dict[tuple, dict] = {}

        for idx in top_indices:
            score = float(mean_scores[idx])
            if score < _MIN_SCORE:
                continue

            sound_name = self._class_names[idx]
            category, block = get_category_info(sound_name)
            if category == "기타" or block == "기타":
                continue

            key = (category, block)
            if key not in block_results or score > block_results[key]["score"]:
                block_results[key] = {"category": category, "block": block, "score": score}

        if not block_results:
            return None

        return max(block_results.values(), key=lambda x: x["score"])
