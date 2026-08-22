import tensorflow as tf

from config import (
    FFT_LENGTH,
    FRAME_LENGTH,
    FRAME_STEP,
    LOWER_FREQUENCY,
    NUM_MEL_BINS,
    SAMPLE_RATE,
    UPPER_FREQUENCY,
)


def waveform_to_log_mel(
    waveform: tf.Tensor,
) -> tf.Tensor:

    waveform = tf.cast(
        waveform,
        tf.float32,
    )

    stft = tf.signal.stft(
        waveform,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )

    spectrogram = tf.abs(
        stft
    )

    power_spectrogram = tf.square(
        spectrogram
    )

    num_spectrogram_bins = (
        FFT_LENGTH // 2 + 1
    )

    mel_weight_matrix = (
        tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=NUM_MEL_BINS,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=SAMPLE_RATE,
            lower_edge_hertz=LOWER_FREQUENCY,
            upper_edge_hertz=UPPER_FREQUENCY,
        )
    )

    mel_spectrogram = tf.matmul(
        power_spectrogram,
        mel_weight_matrix,
    )

    log_mel = tf.math.log(
        mel_spectrogram + 1e-6
    )

    mean = tf.reduce_mean(
        log_mel
    )

    std = tf.math.reduce_std(
        log_mel
    )

    log_mel = (
        log_mel - mean
    ) / (
        std + 1e-6
    )

    log_mel = tf.expand_dims(
        log_mel,
        axis=-1,
    )

    return log_mel