import json
import random

import librosa
import numpy as np
import tensorflow as tf

from config import (
    DATASET_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SAMPLE_RATE,
    CLIP_SAMPLES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
)

from features import waveform_to_log_mel


# ============================================================
# 경로
# ============================================================

PREPARED_V6_DIR = (
    DATASET_DIR
    / "prepared_v6"
)

AUGMENTED_SIREN_DIR = (
    DATASET_DIR
    / "augmented_v8"
    / "train"
    / "emergency"
    / "siren"
)

MODEL_PATH = (
    MODELS_DIR
    / "emergency_classifier_v8.keras"
)

BEST_MODEL_PATH = (
    MODELS_DIR
    / "emergency_classifier_v8_best.keras"
)

HISTORY_PATH = (
    REPORTS_DIR
    / "training_history_v8.json"
)


# ============================================================
# Seed
# ============================================================

def set_seed() -> None:

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    tf.random.set_seed(
        RANDOM_SEED
    )


# ============================================================
# Audio
# ============================================================

def load_audio_numpy(
    path_bytes,
) -> np.ndarray:

    if hasattr(
        path_bytes,
        "numpy",
    ):

        path_string = (
            path_bytes
            .numpy()
            .decode("utf-8")
        )

    else:

        path_string = (
            path_bytes
            .decode("utf-8")
        )

    audio, _ = librosa.load(
        path_string,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if len(audio) < CLIP_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                CLIP_SAMPLES
                - len(audio),
            ),
            mode="constant",
        )

    else:

        audio = audio[
            :CLIP_SAMPLES
        ]

    return audio.astype(
        np.float32
    )


def load_audio_tf(
    path: tf.Tensor,
) -> tf.Tensor:

    waveform = tf.py_function(
        func=load_audio_numpy,
        inp=[path],
        Tout=tf.float32,
    )

    waveform.set_shape(
        [CLIP_SAMPLES]
    )

    return waveform


# ============================================================
# Online augmentation
# ============================================================

def augment_audio(
    waveform: tf.Tensor,
) -> tf.Tensor:

    gain = tf.random.uniform(
        shape=[],
        minval=0.80,
        maxval=1.15,
    )

    waveform = (
        waveform * gain
    )

    shift = tf.random.uniform(
        shape=[],
        minval=-800,
        maxval=801,
        dtype=tf.int32,
    )

    waveform = tf.roll(
        waveform,
        shift=shift,
        axis=0,
    )

    noise_level = tf.random.uniform(
        shape=[],
        minval=0.0,
        maxval=0.005,
    )

    noise = tf.random.normal(
        shape=tf.shape(
            waveform
        ),
        mean=0.0,
        stddev=noise_level,
        dtype=tf.float32,
    )

    waveform = (
        waveform + noise
    )

    waveform = tf.clip_by_value(
        waveform,
        -1.0,
        1.0,
    )

    return waveform


# ============================================================
# Feature
# ============================================================

def decode_file(
    path: tf.Tensor,
    label: tf.Tensor,
    training: bool,
):

    waveform = load_audio_tf(
        path
    )

    if training:

        waveform = augment_audio(
            waveform
        )

    feature = waveform_to_log_mel(
        waveform
    )

    label = tf.cast(
        label,
        tf.float32,
    )

    return (
        feature,
        label,
    )


# ============================================================
# Split 수집
# ============================================================

def collect_split(
    split: str,
):

    emergency_dir = (
        PREPARED_V6_DIR
        / split
        / "emergency"
    )

    normal_dir = (
        PREPARED_V6_DIR
        / split
        / "normal"
    )

    if not emergency_dir.exists():

        raise FileNotFoundError(
            f"폴더 없음: "
            f"{emergency_dir}"
        )

    if not normal_dir.exists():

        raise FileNotFoundError(
            f"폴더 없음: "
            f"{normal_dir}"
        )

    emergency_files = sorted(
        emergency_dir.glob(
            "*.wav"
        )
    )

    normal_files = sorted(
        normal_dir.glob(
            "*.wav"
        )
    )

    # ========================================================
    # Train인 경우 siren 증강 데이터 추가
    # ========================================================

    augmented_siren_files = []

    if split == "train":

        if not AUGMENTED_SIREN_DIR.exists():

            raise FileNotFoundError(
                "Siren 증강 폴더가 없습니다:\n"
                f"{AUGMENTED_SIREN_DIR}\n\n"
                "먼저 augment_siren_v8.py를 실행하세요."
            )

        augmented_siren_files = sorted(
            AUGMENTED_SIREN_DIR.glob(
                "*.wav"
            )
        )

        if not augmented_siren_files:

            raise RuntimeError(
                "Siren 증강 WAV가 없습니다."
            )

        emergency_files = (
            emergency_files
            + augmented_siren_files
        )

    if not emergency_files:

        raise RuntimeError(
            f"{split} emergency 없음"
        )

    if not normal_files:

        raise RuntimeError(
            f"{split} normal 없음"
        )

    paths = (
        emergency_files
        + normal_files
    )

    labels = (
        [1]
        * len(emergency_files)
        +
        [0]
        * len(normal_files)
    )

    print()
    print(
        f"{split:10s}"
    )

    print(
        f"  emergency 전체 : "
        f"{len(emergency_files)}"
    )

    print(
        f"  normal          : "
        f"{len(normal_files)}"
    )

    if split == "train":

        print(
            f"  그중 siren 증강 : "
            f"{len(augmented_siren_files)}"
        )

    print(
        f"  전체            : "
        f"{len(paths)}"
    )

    return (
        [
            str(path)
            for path in paths
        ],
        labels,
    )


# ============================================================
# Dataset
# ============================================================

def make_dataset(
    split: str,
    training: bool,
) -> tf.data.Dataset:

    paths, labels = collect_split(
        split
    )

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                paths,
                labels,
            )
        )
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda path, label:
            decode_file(
                path,
                label,
                training,
            ),
        num_parallel_calls=(
            tf.data.AUTOTUNE
        ),
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# Model
# ============================================================

def build_model(
    input_shape,
) -> tf.keras.Model:

    inputs = tf.keras.Input(
        shape=input_shape,
        name="log_mel",
    )

    # --------------------------------------------------------
    # Block 1
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        filters=12,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(inputs)

    x = (
        tf.keras.layers
        .BatchNormalization()
        (x)
    )

    x = (
        tf.keras.layers
        .ReLU()
        (x)
    )

    x = (
        tf.keras.layers
        .MaxPooling2D(
            pool_size=2
        )
        (x)
    )

    # --------------------------------------------------------
    # Block 2
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        filters=24,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(x)

    x = (
        tf.keras.layers
        .BatchNormalization()
        (x)
    )

    x = (
        tf.keras.layers
        .ReLU()
        (x)
    )

    x = (
        tf.keras.layers
        .MaxPooling2D(
            pool_size=2
        )
        (x)
    )

    # --------------------------------------------------------
    # Block 3
    # --------------------------------------------------------

    x = (
        tf.keras.layers
        .DepthwiseConv2D(
            kernel_size=3,
            padding="same",
            use_bias=False,
        )
        (x)
    )

    x = (
        tf.keras.layers
        .BatchNormalization()
        (x)
    )

    x = (
        tf.keras.layers
        .ReLU()
        (x)
    )

    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=1,
        padding="same",
        use_bias=False,
    )(x)

    x = (
        tf.keras.layers
        .BatchNormalization()
        (x)
    )

    x = (
        tf.keras.layers
        .ReLU()
        (x)
    )

    # --------------------------------------------------------
    # Head
    # --------------------------------------------------------

    x = (
        tf.keras.layers
        .GlobalAveragePooling2D()
        (x)
    )

    x = tf.keras.layers.Dense(
        units=24,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(
        rate=0.25
    )(x)

    outputs = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        name="emergency_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="hearing_emergency_classifier_v8",
    )

    model.compile(
        optimizer=(
            tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE,
            )
        ),

        loss=(
            tf.keras.losses
            .BinaryCrossentropy()
        ),

        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            ),

            tf.keras.metrics.AUC(
                name="auc"
            ),

            tf.keras.metrics.AUC(
                curve="PR",
                name="pr_auc",
            ),
        ],
    )

    return model


# ============================================================
# Main
# ============================================================

def main() -> None:

    set_seed()

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 85)
    print(
        "HEAR:ING Hardware AI "
        "- TRAIN V8"
    )
    print("=" * 85)

    print()
    print(
        "V8 변경:"
    )

    print(
        "- prepared_v6 유지"
    )

    print(
        "- siren offline augmentation 추가"
    )

    print(
        "- class_weight 사용 안 함"
    )

    print(
        "- 모델 구조 V7과 동일"
    )

    print(
        "- final_test 사용 안 함"
    )

    # ========================================================
    # Dataset
    # ========================================================

    print()
    print("=" * 85)
    print("Dataset")
    print("=" * 85)

    train_dataset = make_dataset(
        "train",
        training=True,
    )

    val_dataset = make_dataset(
        "val",
        training=False,
    )

    sample_features, _ = next(
        iter(train_dataset)
    )

    input_shape = tuple(
        sample_features.shape[1:]
    )

    print()
    print(
        f"CNN 입력 shape: "
        f"{input_shape}"
    )

    # ========================================================
    # Model
    # ========================================================

    model = build_model(
        input_shape
    )

    model.summary()

    # ========================================================
    # Callbacks
    # ========================================================

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                BEST_MODEL_PATH
            ),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ========================================================
    # Train
    # ========================================================

    print()
    print("=" * 85)
    print("V8 학습 시작")
    print("=" * 85)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # ========================================================
    # Save
    # ========================================================

    model.save(
        MODEL_PATH
    )

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                key: [
                    float(value)
                    for value in values
                ]
                for key, values
                in history.history.items()
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Validation 0.5 참고
    # ========================================================

    print()
    print("=" * 85)
    print(
        "Validation 참고 결과 "
        "(threshold=0.5)"
    )
    print("=" * 85)

    results = model.evaluate(
        val_dataset,
        return_dict=True,
        verbose=0,
    )

    for key, value in results.items():

        print(
            f"{key}: "
            f"{value:.4f}"
        )

    print()
    print("=" * 85)
    print("V8 학습 완료")
    print("=" * 85)

    print(
        f"최종 모델:\n"
        f"{MODEL_PATH}"
    )

    print(
        f"BEST 모델:\n"
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"학습 기록:\n"
        f"{HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()