import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import (
    MODELS_DIR,
    REPORTS_DIR,
)

from train_v8 import make_dataset


MODEL_PATH = (
    MODELS_DIR
    / "emergency_classifier_v8_best.keras"
)

OUTPUT_PATH = (
    REPORTS_DIR
    / "threshold_v8.json"
)

MIN_RECALL = 0.85


def evaluate_threshold(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[
            0,
            1,
        ],
    )

    return {
        "threshold":
            float(threshold),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "confusion_matrix":
            matrix.tolist(),
    }


def main():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"V8 모델 없음:\n"
            f"{MODEL_PATH}"
        )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 85)
    print(
        "HEAR:ING V8 "
        "Validation Threshold Search"
    )
    print("=" * 85)

    print()
    print(
        "final_test는 사용하지 않습니다."
    )

    model = (
        tf.keras.models.load_model(
            MODEL_PATH
        )
    )

    val_dataset = make_dataset(
        "val",
        training=False,
    )

    y_true = []
    probabilities = []

    print()
    print(
        "Validation 추론 중..."
    )

    for features, labels in val_dataset:

        scores = (
            model.predict(
                features,
                verbose=0,
            )
            .reshape(-1)
        )

        y_true.extend(
            labels.numpy()
            .astype(int)
            .tolist()
        )

        probabilities.extend(
            scores.tolist()
        )

    y_true = np.asarray(
        y_true,
        dtype=np.int32,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    thresholds = np.arange(
        0.025,
        0.801,
        0.025,
    )

    results = []

    print()
    print("=" * 85)
    print("Threshold 비교")
    print("=" * 85)

    print(
        "Threshold | Accuracy | "
        "Precision | Recall | F1"
    )

    print("-" * 85)

    for threshold in thresholds:

        result = evaluate_threshold(
            y_true,
            probabilities,
            float(threshold),
        )

        results.append(
            result
        )

        print(
            f"{threshold:9.3f} | "
            f"{result['accuracy']:8.3f} | "
            f"{result['precision']:9.3f} | "
            f"{result['recall']:6.3f} | "
            f"{result['f1']:6.3f}"
        )

    candidates = [
        result
        for result in results
        if result["recall"] >= MIN_RECALL
    ]

    if candidates:

        recommended = max(
            candidates,
            key=lambda item:
                item["f1"],
        )

        reason = (
            "recall >= 0.85 중 "
            "F1 최고"
        )

    else:

        recommended = max(
            results,
            key=lambda item:
                item["f1"],
        )

        reason = (
            "recall >= 0.85를 "
            "만족하지 못해 F1 최고 선택"
        )

    print()
    print("=" * 85)
    print("추천 Threshold")
    print("=" * 85)

    print(
        f"threshold : "
        f"{recommended['threshold']:.3f}"
    )

    print(
        f"accuracy  : "
        f"{recommended['accuracy']:.4f}"
    )

    print(
        f"precision : "
        f"{recommended['precision']:.4f}"
    )

    print(
        f"recall    : "
        f"{recommended['recall']:.4f}"
    )

    print(
        f"f1        : "
        f"{recommended['f1']:.4f}"
    )

    print()
    print(
        f"선택 이유: "
        f"{reason}"
    )

    print()
    print(
        "혼동행렬:"
    )

    print(
        np.asarray(
            recommended[
                "confusion_matrix"
            ]
        )
    )

    output = {
        "model":
            str(MODEL_PATH),

        "selection_dataset":
            "validation",

        "minimum_recall":
            MIN_RECALL,

        "recommended_threshold":
            recommended[
                "threshold"
            ],

        "recommended_metrics":
            recommended,

        "reason":
            reason,

        "all_threshold_results":
            results,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"저장:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()