import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import (
    MODELS_DIR,
    REPORTS_DIR,
)

from train_v8 import make_dataset


# ============================================================
# 경로
# ============================================================

MODEL_PATH = (
    MODELS_DIR
    / "emergency_classifier_v8_best.keras"
)

THRESHOLD_PATH = (
    REPORTS_DIR
    / "threshold_v8.json"
)

REPORT_PATH = (
    REPORTS_DIR
    / "evaluation_v8_final.json"
)


# ============================================================
# 최종 목표
# ============================================================

TARGET_PRECISION = 0.75
TARGET_RECALL = 0.75
TARGET_F1 = 0.75


# ============================================================
# Main
# ============================================================

def main() -> None:

    # ========================================================
    # 파일 확인
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"V8 BEST 모델이 없습니다:\n"
            f"{MODEL_PATH}"
        )

    if not THRESHOLD_PATH.exists():

        raise FileNotFoundError(
            f"V8 threshold 파일이 없습니다:\n"
            f"{THRESHOLD_PATH}"
        )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 85)
    print(
        "HEAR:ING Hardware AI "
        "- V8 FINAL TEST"
    )
    print("=" * 85)

    print()
    print("주의:")
    print(
        "- validation에서 결정한 threshold를 그대로 사용합니다."
    )
    print(
        "- final_test에서 threshold를 다시 탐색하지 않습니다."
    )

    # ========================================================
    # Threshold
    # ========================================================

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        threshold_data = json.load(
            file
        )

    threshold = float(
        threshold_data[
            "recommended_threshold"
        ]
    )

    print()
    print(
        f"Validation threshold: "
        f"{threshold:.3f}"
    )

    print(
        f"모델: "
        f"{MODEL_PATH}"
    )

    # ========================================================
    # Model
    # ========================================================

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # ========================================================
    # Final Test Dataset
    # ========================================================

    final_test_dataset = make_dataset(
        "final_test",
        training=False,
    )

    y_true = []
    probabilities = []

    print()
    print(
        "Final Test 추론 중..."
    )

    for features, labels in final_test_dataset:

        scores = model.predict(
            features,
            verbose=0,
        ).reshape(-1)

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

    # ========================================================
    # Prediction
    # ========================================================

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int32
    )

    # ========================================================
    # Metrics
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[
            0,
            1,
        ],
    )

    # ========================================================
    # Confusion Matrix
    # ========================================================

    tn = int(
        matrix[0][0]
    )

    fp = int(
        matrix[0][1]
    )

    fn = int(
        matrix[1][0]
    )

    tp = int(
        matrix[1][1]
    )

    normal_total = (
        tn + fp
    )

    emergency_total = (
        tp + fn
    )

    false_alarm_rate = (
        fp / normal_total
        if normal_total > 0
        else 0.0
    )

    emergency_miss_rate = (
        fn / emergency_total
        if emergency_total > 0
        else 0.0
    )

    # ========================================================
    # 출력
    # ========================================================

    print()
    print("=" * 85)
    print(
        "V8 FINAL TEST 결과"
    )
    print("=" * 85)

    print()
    print(
        f"Final Test 전체 : "
        f"{len(y_true)}"
    )

    print(
        f"normal          : "
        f"{np.sum(y_true == 0)}"
    )

    print(
        f"emergency       : "
        f"{np.sum(y_true == 1)}"
    )

    print()
    print(
        f"Threshold : "
        f"{threshold:.3f}"
    )

    print(
        f"Accuracy  : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"F1        : "
        f"{f1:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{pr_auc:.4f}"
    )

    print()
    print("혼동행렬")

    print(
        matrix
    )

    print()

    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "normal",
                "emergency",
            ],
            digits=4,
            zero_division=0,
        )
    )

    # ========================================================
    # 상세
    # ========================================================

    print("=" * 85)
    print(
        "긴급 감지 상세"
    )
    print("=" * 85)

    print(
        f"TN "
        f"(일반 → 일반) : "
        f"{tn}"
    )

    print(
        f"FP "
        f"(일반 → 긴급) : "
        f"{fp}"
    )

    print(
        f"FN "
        f"(긴급 → 일반) : "
        f"{fn}"
    )

    print(
        f"TP "
        f"(긴급 → 긴급) : "
        f"{tp}"
    )

    print()

    print(
        f"일반 오탐률 : "
        f"{false_alarm_rate * 100:.2f}%"
    )

    print(
        f"긴급 누락률 : "
        f"{emergency_miss_rate * 100:.2f}%"
    )

    # ========================================================
    # 목표 판정
    # ========================================================

    precision_pass = (
        precision
        >= TARGET_PRECISION
    )

    recall_pass = (
        recall
        >= TARGET_RECALL
    )

    f1_pass = (
        f1
        >= TARGET_F1
    )

    overall_pass = (
        precision_pass
        and recall_pass
        and f1_pass
    )

    print()
    print("=" * 85)
    print(
        "최종 목표 기준"
    )
    print("=" * 85)

    print(
        f"Precision >= "
        f"{TARGET_PRECISION:.2f} : "
        f"{'PASS' if precision_pass else 'FAIL'}"
    )

    print(
        f"Recall    >= "
        f"{TARGET_RECALL:.2f} : "
        f"{'PASS' if recall_pass else 'FAIL'}"
    )

    print(
        f"F1        >= "
        f"{TARGET_F1:.2f} : "
        f"{'PASS' if f1_pass else 'FAIL'}"
    )

    print()

    print(
        f"최종 판정: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    # ========================================================
    # JSON 저장
    # ========================================================

    output = {
        "model":
            str(
                MODEL_PATH
            ),

        "dataset":
            "final_test",

        "threshold_source":
            "validation",

        "threshold":
            threshold,

        "samples": {
            "total":
                int(
                    len(y_true)
                ),

            "normal":
                int(
                    np.sum(
                        y_true == 0
                    )
                ),

            "emergency":
                int(
                    np.sum(
                        y_true == 1
                    )
                ),
        },

        "metrics": {
            "accuracy":
                float(
                    accuracy
                ),

            "precision":
                float(
                    precision
                ),

            "recall":
                float(
                    recall
                ),

            "f1":
                float(
                    f1
                ),

            "roc_auc":
                float(
                    roc_auc
                ),

            "pr_auc":
                float(
                    pr_auc
                ),
        },

        "confusion_matrix":
            matrix.tolist(),

        "counts": {
            "true_negative":
                tn,

            "false_positive":
                fp,

            "false_negative":
                fn,

            "true_positive":
                tp,
        },

        "rates": {
            "false_alarm_rate":
                float(
                    false_alarm_rate
                ),

            "emergency_miss_rate":
                float(
                    emergency_miss_rate
                ),
        },

        "targets": {
            "precision":
                TARGET_PRECISION,

            "recall":
                TARGET_RECALL,

            "f1":
                TARGET_F1,
        },

        "target_pass": {
            "precision":
                precision_pass,

            "recall":
                recall_pass,

            "f1":
                f1_pass,

            "overall":
                overall_pass,
        },
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # 완료
    # ========================================================

    print()
    print(
        f"최종 평가 저장:\n"
        f"{REPORT_PATH}"
    )

    print()
    print("=" * 85)

    if overall_pass:

        print(
            "V8 최종 기준 통과."
        )

        print(
            "다음 단계: "
            "TFLite INT8 양자화"
        )

    else:

        print(
            "V8 최종 기준 미통과."
        )

        print(
            "final_test 기준으로 "
            "threshold를 다시 조정하지 마세요."
        )

        print(
            "다음 개선은 train/validation "
            "기준으로 진행합니다."
        )

    print("=" * 85)


if __name__ == "__main__":
    main()