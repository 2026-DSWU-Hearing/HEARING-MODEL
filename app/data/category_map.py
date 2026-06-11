CATEGORY_MAP = {
    # 긴급
    "Fire": ("긴급", "화재 경보"),
    "Crackle": ("긴급", "화재 경보"),
    "Smoke detector, smoke alarm": ("긴급", "화재 경보"),
    "Fire alarm": ("긴급", "화재 경보"),

    "Siren": ("긴급", "사이렌"),
    "Civil defense siren": ("긴급", "사이렌"),

    "Buzzer": ("긴급", "경보음"),
    "Alarm": ("긴급", "경보음"),

    "Emergency vehicle": ("긴급", "응급차량"),
    "Police car (siren)": ("긴급", "응급차량"),
    "Ambulance (siren)": ("긴급", "응급차량"),
    "Fire engine, fire truck (siren)": ("긴급", "응급차량"),

    "Explosion": ("긴급", "폭발·파열음"),
    "Fireworks": ("긴급", "폭발·파열음"),
    "Firecracker": ("긴급", "폭발·파열음"),
    "Burst, pop": ("긴급", "폭발·파열음"),
    "Boom": ("긴급", "폭발·파열음"),

    "Shatter": ("긴급", "충돌·파손음"),
    "Thump, thud": ("긴급", "충돌·파손음"),
    "Bang": ("긴급", "충돌·파손음"),
    "Smash, crash": ("긴급", "충돌·파손음"),
    "Breaking": ("긴급", "충돌·파손음"),
    "Crushing": ("긴급", "충돌·파손음"),

    # 교통
    "Car alarm": ("교통", "자동차 경고음"),
    "Reversing beeps": ("교통", "자동차 경고음"),

    "Skidding": ("교통", "급정거·마찰음"),
    "Tire squeal": ("교통", "급정거·마찰음"),

    "Vehicle horn, car horn, honking": ("교통", "경적"),
    "Air horn, truck horn": ("교통", "경적"),

    "Train whistle": ("교통", "기차"),
    "Train horn": ("교통", "기차"),
    "Train wheels squealing": ("교통", "기차"),
    "Rail transport": ("교통", "기차"),
    "Train": ("교통", "기차"),

    "Motor vehicle (road)": ("교통", "차량 주행음"),
    "Car": ("교통", "차량 주행음"),
    "Truck": ("교통", "차량 주행음"),
    "Air brake": ("교통", "차량 주행음"),
    "Traffic noise, roadway noise": ("교통", "차량 주행음"),
    "Vehicle":("교통", "차량 주행음"),

    "Motorcycle": ("교통", "오토바이"),

    "Aircraft": ("교통", "항공기"),
    "Aircraft engine": ("교통", "항공기"),
    "Jet engine": ("교통", "항공기"),
    "Helicopter": ("교통", "항공기"),

    "Engine": ("교통", "엔진"),
    "Engine knocking": ("교통", "엔진"),
    "Engine starting": ("교통", "엔진"),
    "Accelerating, revving, vroom": ("교통", "엔진"),

    "Bicycle": ("교통", "기타 이동수단"),
    "Elevator": ("교통", "기타 이동수단"),
    "Escalator": ("교통", "기타 이동수단"),

    # 사람
    "Shout": ("사람", "비명"),
    "Bellow": ("사람", "비명"),
    "Yell": ("사람", "비명"),
    "Children shouting": ("사람", "비명"),
    "Screaming": ("사람", "비명"),

    "Crying, sobbing": ("사람", "울음"),
    "Baby cry, infant cry": ("사람", "울음"),

    "Wheeze": ("사람", "호흡·기침"),
    "Gasp": ("사람", "호흡·기침"),
    "Pant": ("사람", "호흡·기침"),
    "Cough": ("사람", "호흡·기침"),
    "Breathing": ("사람", "호흡·기침"),
    "Snoring": ("사람", "호흡·기침"),

    "Wail, moan": ("사람", "신음"),
    "Groan": ("사람", "신음"),

    "Speech": ("사람", "음성"),
    "Conversation": ("사람", "음성"),
    "Narration, monologue": ("사람", "음성"),
    "Child speech, kid speaking": ("사람", "음성"),
    "Male speech, man speaking": ("사람", "음성"),
    "Female speech, woman speaking": ("사람", "음성"),

    "Babbling": ("사람", "아기 옹알이"),

    "Laughter": ("사람", "웃음"),
    "Giggle": ("사람", "웃음"),
    "Chuckle, chortle": ("사람", "웃음"),
    "Snicker": ("사람", "웃음"),

    "Singing": ("사람", "노래"),
    "Choir": ("사람", "노래"),
    "Yodeling": ("사람", "노래"),
    "Chant": ("사람", "노래"),

    "Crowd": ("사람", "군중"),
    "Footsteps": ("사람", "발걸음"),

    # 생활음
    "Vacuum cleaner": ("생활음", "가전제품"),
    "Washing machine": ("생활음", "가전제품"),

    "Toilet flush": ("생활음", "욕실 물 소리"),
    "Shower": ("생활음", "욕실 물 소리"),
    "Sink (filling or washing)": ("생활음", "욕실 물 소리"),

    "Splash, splatter": ("생활음", "물 튀는 소리"),
    "Gush": ("생활음", "물 튀는 소리"),
    "Spray": ("생활음", "스프레이"),

    "Printer": ("생활음", "사무기기"),
    "Typing": ("생활음", "사무기기"),
    "Computer keyboard": ("생활음", "사무기기"),

    "Clicking": ("생활음", "클릭음"),
    "Tick": ("생활음", "클릭음"),
    "Tap": ("생활음", "클릭음"),

    "Swish": ("생활음", "휙 소리"),
    "Whoosh, swoosh, swish": ("생활음", "휙 소리"),
    "Rustle": ("생활음", "바스락"),

    "Beep, bleep": ("생활음", "전자 알림음"),
    "Ping": ("생활음", "전자 알림음"),
    "Ding": ("생활음", "전자 알림음"),

    "Slam": ("생활음", "문 닫힘 소리"),
    "Knock": ("생활음", "노크 소리"),
    "Keys jangling": ("생활음", "열쇠 소리"),

    "Crack": ("생활음", "충돌·파손음"),
    "Clang": ("생활음", "충돌·파손음"),
    "Clatter": ("생활음", "충돌·파손음"),
    "Rumble": ("생활음", "충돌·파손음"),
    "Crunch": ("생활음", "충돌·파손음"),
    "Vibration": ("생활음", "충돌·파손음"),

    "Tools": ("생활음", "공구"),
    "Hammer": ("생활음", "공구"),
    "Jackhammer": ("생활음", "공구"),
    "Sawing": ("생활음", "공구"),
    "Power tool": ("생활음", "공구"),
    "Drill": ("생활음", "공구"),
    "Lawn mower": ("생활음", "공구"),
    "Chainsaw": ("생활음", "공구"),

    "Office noise": ("생활음", "실내 배경음"),
    "Restaurant noise": ("생활음", "실내 배경음"),

    "Slap, smack": ("생활음", "타격음"),
    "Whack, thwack": ("생활음", "타격음"),
    "Whip": ("생활음", "타격음"),

    "Scrape": ("생활음", "마찰음"),
    "Squeal": ("생활음", "마찰음"),
    "Creak": ("생활음", "마찰음"),
    "Squeak": ("생활음", "마찰음"),

    "Noise": ("생활음", "소음"),
    "Environmental noise": ("생활음", "소음"),
    "Cacophony": ("생활음", "소음"),
    "White noise": ("생활음", "소음"),

    "Air conditioning": ("생활음", "냉난방·기계음"),
    "Fan": ("생활음", "냉난방·기계음"),
    "Hum": ("생활음", "냉난방·기계음"),

    "Steam": ("생활음", "증기"),

    # 자연
    "Wind": ("자연", "바람·비"),
    "Rain": ("자연", "바람·비"),
    "Thunderstorm": ("자연", "천둥"),
    "Thunder": ("자연", "천둥"),
    "Leaves rustling": ("자연", "나뭇잎 소리"),
    "Eruption": ("자연", "화산 분출음"),
    "Ocean": ("자연", "물가"),
    "Waves, surf": ("자연", "물가"),
    "Stream": ("자연", "물가"),

    # 동물
    "Cat": ("동물", "고양이"),
    "Meow": ("동물", "고양이"),
    "Caterwaul": ("동물", "고양이"),

    "Bark": ("동물", "개"),
    "Howl": ("동물", "개"),
    "Canidae, dogs, wolves": ("동물", "개"),

    "Bird": ("동물", "새"),
    "Bird vocalization, bird call, bird song": ("동물", "새"),
    "Chirp, tweet": ("동물", "새"),

    "Domestic animals, pets": ("동물", "가축"),
    "Livestock, farm animals, working animals": ("동물", "가축"),

    "Hiss": ("동물", "뱀"),
    "Snake": ("동물", "뱀"),

    "Bee, wasp, etc.": ("동물", "벌레"),
    "Insect": ("동물", "벌레"),

    "Growling": ("동물", "맹수·야생동물"),
    "Roar": ("동물", "맹수·야생동물"),
    "Wild animals": ("동물", "맹수·야생동물"),
    "Roaring cats (lions, tigers)": ("동물", "맹수·야생동물"),

    # 주방
    "Boiling": ("주방", "끓는 소리"),
    "Dishes, pots, and pans": ("주방", "식기"),
    "Cutlery, silverware": ("주방", "식기"),
    "Frying (food)": ("주방", "조리"),
    "Microwave oven": ("주방", "조리"),
    "Blender": ("주방", "조리"),

    # 음악
    "Music": ("음악", "대중 음악"),
    "Electronic music": ("음악", "대중 음악"),
    "Piano": ("음악", "피아노"),
    "Guitar": ("음악", "현악기"),
    "Violin, fiddle": ("음악", "현악기"),
    "Drum": ("음악", "드럼"),
    "Jazz": ("음악", "재즈"),
    "Classical music": ("음악", "클래식"),
    "Orchestra": ("음악", "클래식"),
}


def get_category_info(class_name: str):
    return CATEGORY_MAP.get(class_name, ("기타", "기타"))
