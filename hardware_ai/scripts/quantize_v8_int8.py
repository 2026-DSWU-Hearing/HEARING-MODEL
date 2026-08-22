import json
import shutil

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from config import (
    MODELS_DIR,
    REPORTS_DIR,
)

from train_v8 import make_dataset


# ============================================================
# 경로
# ============================================================

KERAS_MODEL_PATH = (
    MODELS_DIR
    / "emergency_classifier_v8_best.keras"
)

SAVED_MODEL_DIR = (
    MODELS_DIR
    / "emergency_classifier_v8_savedmodel"
)

TFLITE_MODEL_PATH = (
    MODELS_DIR
    / "emergency_classifier_v8_int8.tflite"
)

THRESHOLD_PATH = (
    REPORTS_DIR
    / "threshold_v8.json"
)

REPORT_PATH = (
    REPORTS_DIR
    / "quantization_v8.json"
)


# ============================================================
# Representative Dataset
# ============================================================

REPRESENTATIVE_BATCHES = 100


# ============================================================
# SavedModel 생성
# ============================================================

def export_saved_model() -> None:

    if not KERAS_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Keras 모델 없음:\n"
            f"{KERAS_MODEL_PATH}"
        )

    print()
    print("=" * 80)
    print("Keras → SavedModel")
    print("=" * 80)

    print()
    print(
        f"Keras 모델 로드:\n"
        f"{KERAS_MODEL_PATH}"
    )

    model = tf.keras.models.load_model(
        KERAS_MODEL_PATH
    )

    # 기존 SavedModel 제거
    if SAVED_MODEL_DIR.exists():

        try:

            shutil.rmtree(
                SAVED_MODEL_DIR
            )

        except PermissionError:

            raise PermissionError(
                "기존 SavedModel 폴더를 삭제하지 못했습니다.\n"
                f"{SAVED_MODEL_DIR}\n\n"
                "VS Code 탐색기/OneDrive가 폴더를 잡고 있는지 확인하세요."
            )

    print()
    print(
        f"SavedModel export:\n"
        f"{SAVED_MODEL_DIR}"
    )

    # --------------------------------------------------------
    # Keras 3에서는 model.save()로 SavedModel 저장하지 않고
    # model.export() 사용
    # --------------------------------------------------------

    model.export(
        SAVED_MODEL_DIR
    )

    saved_model_pb = (
        SAVED_MODEL_DIR
        / "saved_model.pb"
    )

    if not saved_model_pb.exists():

        raise RuntimeError(
            "SavedModel 생성 실패:\n"
            f"{saved_model_pb}"
        )

    print()
    print(
        "SavedModel 생성 완료"
    )


# ============================================================
# Representative dataset
# ============================================================

def representative_dataset():
    """
    prepared_v6/train에서
    실제 log-mel feature를 calibration에 사용.

    주의:
    training=False로 augmentation 없이 대표 분포만 사용한다.
    """

    train_dataset = make_dataset(
        "train",
        training=False,
    )

    batch_count = 0
    sample_count = 0

    for features, _ in train_dataset:

        features_np = (
            features.numpy()
            .astype(np.float32)
        )

        for index in range(
            len(features_np)
        ):

            sample = features_np[
                index:index + 1
            ]

            yield [
                sample
            ]

            sample_count += 1

        batch_count += 1

        if (
            batch_count
            >= REPRESENTATIVE_BATCHES
        ):
            break

    print(
        f"Representative samples 사용: "
        f"{sample_count}"
    )


# ============================================================
# INT8 변환
# ============================================================

def quantize_model() -> None:

    if not SAVED_MODEL_DIR.exists():

        raise FileNotFoundError(
            f"SavedModel 없음:\n"
            f"{SAVED_MODEL_DIR}"
        )

    print()
    print("=" * 80)
    print("SavedModel → TFLite INT8")
    print("=" * 80)

    converter = (
        tf.lite.TFLiteConverter
        .from_saved_model(
            str(
                SAVED_MODEL_DIR
            )
        )
    )

    # 기본 최적화
    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    # Calibration
    converter.representative_dataset = (
        representative_dataset
    )

    # 완전 INT8 op만 허용
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    # 입출력도 INT8
    converter.inference_input_type = (
        tf.int8
    )

    converter.inference_output_type = (
        tf.int8
    )

    print()
    print(
        "INT8 변환 중..."
    )

    tflite_model = converter.convert()

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        TFLITE_MODEL_PATH,
        "wb",
    ) as file:

        file.write(
            tflite_model
        )

    print()
    print(
        f"INT8 모델 저장:\n"
        f"{TFLITE_MODEL_PATH}"
    )

    # ========================================================
    # 크기
    # ========================================================

    keras_size = (
        KERAS_MODEL_PATH
        .stat()
        .st_size
    )

    int8_size = (
        TFLITE_MODEL_PATH
        .stat()
        .st_size
    )

    reduction = (
        1.0
        - (
            int8_size
            / keras_size
        )
    )

    print()
    print("=" * 80)
    print("모델 크기")
    print("=" * 80)

    print(
        f"Keras : "
        f"{keras_size / 1024:.2f} KB"
    )

    print(
        f"INT8  : "
        f"{int8_size / 1024:.2f} KB"
    )

    print(
        f"감소율: "
        f"{reduction * 100:.2f}%"
    )


# ============================================================
# Interpreter 정보
# ============================================================

def inspect_int8_model():

    interpreter = (
        tf.lite.Interpreter(
            model_path=str(
                TFLITE_MODEL_PATH
            )
        )
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter
        .get_input_details()
    )

    output_details = (
        interpreter
        .get_output_details()
    )

    input_info = input_details[0]
    output_info = output_details[0]

    print()
    print("=" * 80)
    print("TFLite Tensor 정보")
    print("=" * 80)

    print()
    print("Input")

    print(
        f"dtype      : "
        f"{input_info['dtype']}"
    )

    print(
        f"shape      : "
        f"{input_info['shape']}"
    )

    print(
        f"quantization: "
        f"{input_info['quantization']}"
    )

    print()
    print("Output")

    print(
        f"dtype      : "
        f"{output_info['dtype']}"
    )

    print(
        f"shape      : "
        f"{output_info['shape']}"
    )

    print(
        f"quantization: "
        f"{output_info['quantization']}"
    )

    if (
        input_info["dtype"]
        is not np.int8
    ):

        raise RuntimeError(
            "입력이 INT8이 아닙니다."
        )

    if (
        output_info["dtype"]
        is not np.int8
    ):

        raise RuntimeError(
            "출력이 INT8이 아닙니다."
        )

    return (
        interpreter,
        input_info,
        output_info,
    )


# ============================================================
# Float 추론
# ============================================================

def run_float_inference(
    dataset,
):

    model = (
        tf.keras.models.load_model(
            KERAS_MODEL_PATH
        )
    )

    y_true = []
    probabilities = []

    for features, labels in dataset:

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

    return (
        np.asarray(
            y_true,
            dtype=np.int32,
        ),
        np.asarray(
            probabilities,
            dtype=np.float32,
        ),
    )


# ============================================================
# INT8 추론
# ============================================================

def run_int8_inference(
    dataset,
):

    (
        interpreter,
        input_info,
        output_info,
    ) = inspect_int8_model()

    input_scale = float(
        input_info[
            "quantization"
        ][0]
    )

    input_zero_point = int(
        input_info[
            "quantization"
        ][1]
    )

    output_scale = float(
        output_info[
            "quantization"
        ][0]
    )

    output_zero_point = int(
        output_info[
            "quantization"
        ][1]
    )

    if input_scale == 0:

        raise RuntimeError(
            "입력 quantization scale이 0입니다."
        )

    if output_scale == 0:

        raise RuntimeError(
            "출력 quantization scale이 0입니다."
        )

    y_true = []
    probabilities = []

    for features, labels in dataset:

        features_np = (
            features.numpy()
            .astype(np.float32)
        )

        labels_np = (
            labels.numpy()
            .astype(np.int32)
        )

        for index in range(
            len(features_np)
        ):

            sample = features_np[
                index:index + 1
            ]

            # ------------------------------------------------
            # float → int8
            #
            # q = round(real / scale + zero_point)
            # ------------------------------------------------

            quantized_input = (
                np.round(
                    sample
                    / input_scale
                    + input_zero_point
                )
            )

            quantized_input = (
                np.clip(
                    quantized_input,
                    -128,
                    127,
                )
                .astype(
                    np.int8
                )
            )

            interpreter.set_tensor(
                input_info[
                    "index"
                ],
                quantized_input,
            )

            interpreter.invoke()

            quantized_output = (
                interpreter.get_tensor(
                    output_info[
                        "index"
                    ]
                )
            )

            # ------------------------------------------------
            # int8 → float
            #
            # real = (q - zero_point) * scale
            # ------------------------------------------------

            probability = (
                (
                    quantized_output
                    .astype(
                        np.float32
                    )
                    - output_zero_point
                )
                * output_scale
            )

            probability = float(
                probability.reshape(
                    -1
                )[0]
            )

            y_true.append(
                int(
                    labels_np[index]
                )
            )

            probabilities.append(
                probability
            )

    return (
        np.asarray(
            y_true,
            dtype=np.int32,
        ),
        np.asarray(
            probabilities,
            dtype=np.float32,
        ),
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int32
    )

    return {
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
    }


# ============================================================
# Float vs INT8
# ============================================================

def compare_models():

    if not THRESHOLD_PATH.exists():

        raise FileNotFoundError(
            f"Threshold 없음:\n"
            f"{THRESHOLD_PATH}"
        )

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        threshold_data = (
            json.load(file)
        )

    threshold = float(
        threshold_data[
            "recommended_threshold"
        ]
    )

    print()
    print("=" * 80)
    print("Float32 vs INT8")
    print("=" * 80)

    print(
        f"Threshold: "
        f"{threshold:.3f}"
    )

    # final_test 사용하지 않음
    float_dataset = make_dataset(
        "val",
        training=False,
    )

    int8_dataset = make_dataset(
        "val",
        training=False,
    )

    print()
    print(
        "Float32 validation 추론 중..."
    )

    (
        float_y,
        float_probabilities,
    ) = run_float_inference(
        float_dataset
    )

    print()
    print(
        "INT8 validation 추론 중..."
    )

    (
        int8_y,
        int8_probabilities,
    ) = run_int8_inference(
        int8_dataset
    )

    if not np.array_equal(
        float_y,
        int8_y,
    ):

        raise RuntimeError(
            "Float32와 INT8 label 순서가 다릅니다."
        )

    float_metrics = (
        calculate_metrics(
            float_y,
            float_probabilities,
            threshold,
        )
    )

    int8_metrics = (
        calculate_metrics(
            int8_y,
            int8_probabilities,
            threshold,
        )
    )

    print()
    print("=" * 80)
    print("Float32")
    print("=" * 80)

    for key, value in (
        float_metrics.items()
    ):

        print(
            f"{key:10s}: "
            f"{value:.4f}"
        )

    print()
    print("=" * 80)
    print("INT8")
    print("=" * 80)

    for key, value in (
        int8_metrics.items()
    ):

        print(
            f"{key:10s}: "
            f"{value:.4f}"
        )

    print()
    print("=" * 80)
    print("차이 (INT8 - Float32)")
    print("=" * 80)

    differences = {}

    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
    ):

        difference = (
            int8_metrics[key]
            - float_metrics[key]
        )

        differences[key] = float(
            difference
        )

        print(
            f"{key:10s}: "
            f"{difference:+.4f}"
        )

    # ========================================================
    # Report
    # ========================================================

    output = {
        "keras_model":
            str(
                KERAS_MODEL_PATH
            ),

        "saved_model":
            str(
                SAVED_MODEL_DIR
            ),

        "int8_model":
            str(
                TFLITE_MODEL_PATH
            ),

        "threshold":
            threshold,

        "evaluation_dataset":
            "validation",

        "float_metrics":
            float_metrics,

        "int8_metrics":
            int8_metrics,

        "difference":
            differences,

        "keras_size_bytes":
            int(
                KERAS_MODEL_PATH
                .stat()
                .st_size
            ),

        "int8_size_bytes":
            int(
                TFLITE_MODEL_PATH
                .stat()
                .st_size
            ),
    }

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    print()
    print(
        f"Report 저장:\n"
        f"{REPORT_PATH}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "HEAR:ING V8 "
        "Full INT8 Quantization"
    )
    print("=" * 80)

    # 1. Keras → SavedModel
    export_saved_model()

    # 2. SavedModel → TFLite INT8
    quantize_model()

    # 3. INT8 Tensor 확인
    inspect_int8_model()

    # 4. Validation 비교
    compare_models()

    print()
    print("=" * 80)
    print("V8 INT8 양자화 완료")
    print("=" * 80)

    print()
    print(
        f"생성 모델:\n"
        f"{TFLITE_MODEL_PATH}"
    )

    print()
    print(
        "성능 차이가 작으면 "
        "다음 단계는 ESP32용 C 배열 변환입니다."
    )


if __name__ == "__main__":
    main()