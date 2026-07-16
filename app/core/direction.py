from enum import IntEnum


class Direction(IntEnum):
    FRONT = 0
    BACK = 1
    LEFT = 2
    RIGHT = 3
    UNKNOWN = 4


DIRECTION_NAMES = {
    Direction.FRONT: "FRONT",
    Direction.BACK: "BACK",
    Direction.LEFT: "LEFT",
    Direction.RIGHT: "RIGHT",
    Direction.UNKNOWN: "UNKNOWN",
}


def parse_audio_packet(packet: bytes) -> tuple[int, str, bytes]:
    """
    ESP32 패킷 구조

    [0]     : 방향 1바이트
    [1:4]   : 패딩 3바이트
    [4:]    : PCM signed int16 little-endian 오디오

    반환값:
        direction_value: 0~4
        direction_name: FRONT/BACK/LEFT/RIGHT/UNKNOWN
        pcm_audio: 헤더가 제거된 순수 PCM 데이터
    """

    if len(packet) < 4:
        raise ValueError(
            f"패킷 길이가 너무 짧습니다: {len(packet)}바이트"
        )

    raw_direction = packet[0]

    try:
        direction_enum = Direction(raw_direction)
    except ValueError:
        direction_enum = Direction.UNKNOWN

    pcm_audio = packet[4:]

    if not pcm_audio:
        raise ValueError("PCM 오디오 데이터가 없습니다.")

    # int16 샘플은 하나당 2바이트
    if len(pcm_audio) % 2 != 0:
        raise ValueError(
            f"PCM int16 데이터 길이는 짝수여야 합니다: "
            f"{len(pcm_audio)}바이트"
        )

    return (
        int(direction_enum.value),
        DIRECTION_NAMES[direction_enum],
        pcm_audio,
    )