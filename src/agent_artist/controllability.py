"""
Тест управляемости: выполняет ли генератор прямые указания промпта.

Цикл критики почти не улучшает картину, и карта непосильных деталей указывает на
расположение и свет. Здесь это проверяется напрямую, без художника-LLM: промпты с
проверяемыми фактами (что слева, сколько предметов, куда смотрит, откуда свет)
написаны вручную на двух языках, проверяющий по картине отвечает на вопрос по
каждому факту. Кеш — results.json в папке теста.
"""

import statistics
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, artists_by_ids
from .generators import create_image_generator
from .llm import YandexLLM, create_yandex_client
from .runlogs import JsonCache, markdown_table

FACT_CHECK_SYSTEM_PROMPT = (
    "Ты внимательно смотришь на картину и отвечаешь на вопросы о ней.\n"
    "«Слева» и «справа» — с точки зрения зрителя картины. Отвечай только тем, что видно.\n"
    "Ответь ТОЛЬКО JSON-объектом, без пояснений и без markdown.\n"
    'Ответы в том же порядке, что и вопросы, каждый — "да", "нет" или "не видно": '
    '{"answers": ["да", "нет"]}'
)


@dataclass
class Fact:
    kind: str        # placement, count, direction, light, depth
    question: str    # вопрос проверяющему, ответ «да» = указание выполнено


@dataclass
class Task:
    task_id: str
    prompt_ru: str
    prompt_en: str
    facts: list[Fact]


TASKS = [
    Task("ball_cube",
         "Красный шар на левой стороне стола, синий куб на правой стороне стола, белый фон, студийный свет.",
         "A red ball on the left side of a table, a blue cube on the right side of the table, white background, studio light.",
         [Fact("placement", "Красный шар находится левее синего куба?"),
          Fact("count", "На картине ровно один шар и ровно один куб?")]),
    Task("apples",
         "Ровно три жёлтых яблока в ряд на деревянном столе, крупный план.",
         "Exactly three yellow apples in a row on a wooden table, close-up.",
         [Fact("count", "На картине ровно три яблока?"),
          Fact("placement", "Все яблоки стоят в один ряд?")]),
    Task("cat_gaze",
         "Рыжая кошка сидит в центре кадра и смотрит вправо, профиль, однотонный серый фон.",
         "A ginger cat sitting in the center of the frame looking to the right, profile view, plain grey background.",
         [Fact("direction", "Кошка смотрит вправо (её нос направлен к правому краю картины)?"),
          Fact("placement", "Кошка находится в центре кадра?")]),
    Task("sun_house",
         "Солнце в правом верхнем углу неба, маленький дом в левом нижнем углу, зелёное поле.",
         "The sun in the upper right corner of the sky, a small house in the lower left corner, green field.",
         [Fact("placement", "Солнце находится в правой верхней части картины?"),
          Fact("placement", "Дом находится в левой нижней части картины?")]),
    Task("vase_light",
         "Белая ваза на столе, свет падает слева, длинная тень от вазы тянется вправо.",
         "A white vase on a table, light coming from the left, a long shadow of the vase stretching to the right.",
         [Fact("light", "Свет падает слева (освещена левая сторона вазы)?"),
          Fact("light", "Тень от вазы лежит справа от неё?")]),
    Task("car_house",
         "Зелёная машина на переднем плане, красный дом далеко на заднем плане.",
         "A green car in the foreground, a red house far away in the background.",
         [Fact("depth", "Зелёная машина на переднем плане, ближе к зрителю, чем дом?"),
          Fact("count", "На картине ровно одна машина и ровно один дом?")]),
    Task("dogs",
         "Две собаки: большая белая собака слева, маленькая чёрная собака справа, парк.",
         "Two dogs: a large white dog on the left, a small black dog on the right, a park.",
         [Fact("count", "На картине ровно две собаки?"),
          Fact("placement", "Белая собака левее чёрной?"),
          Fact("placement", "Белая собака крупнее чёрной?")]),
    Task("lamp_night",
         "Ночная улица, единственный фонарь в правой части кадра, свет от него падает на мостовую справа, левая часть кадра в темноте.",
         "A night street, a single street lamp on the right side of the frame lighting the pavement on the right, the left side of the frame in darkness.",
         [Fact("placement", "Фонарь находится в правой части картины?"),
          Fact("light", "Левая часть картины темнее правой?")]),
]

ANSWER_SCORES = {"да": 1.0, "нет": 0.0, "не видно": 0.0}
KIND_NAMES = {"placement": "расположение", "count": "количество", "direction": "направление взгляда",
              "light": "свет и тень", "depth": "передний и задний план"}


def run_benchmark(settings: Settings, artist_ids: list[str], repeats: int, root: Path) -> str:
    client = create_yandex_client(settings.yandex_api_key, settings.yandex_folder_id)
    checker = YandexLLM(client, settings.yandex_folder_id, settings.expert_vision_llm_name)
    root.mkdir(parents=True, exist_ok=True)
    cache = JsonCache(root / "results.json")

    results = []   # (artist_label, kind, score)
    for artist in artists_by_ids(artist_ids):
        generator = create_image_generator(artist, settings, client)
        english = artist.prompt_language == "английском"
        label = f"{artist.artist_id} ({'SDXL' if artist.generator == 'local' else 'Yandex ART'}, " \
                f"{'англ.' if english else 'рус.'})"
        for task in TASKS:
            for attempt in range(repeats):
                key = f"{artist.artist_id}_{task.task_id}_{attempt}"
                image_path = root / f"{key}.png"

                def check():
                    print(f"  {key}...")
                    if hasattr(generator, "seed"):
                        generator.seed = settings.seed + attempt   # разные картины у SDXL
                    try:
                        generator.generate(task.prompt_en if english else task.prompt_ru, image_path)
                    except Exception as error:
                        return {"error": str(error), "answers": []}
                    questions = "\n".join(f"{number}. {fact.question}"
                                          for number, fact in enumerate(task.facts, 1))
                    verdict = checker.ask_json(FACT_CHECK_SYSTEM_PROMPT, questions,
                                               image_path=image_path,
                                               required_fields=("answers",), temperature=0.2)
                    return {"error": None,
                            "answers": [str(answer).lower() for answer in verdict["answers"]]}

                entry = cache.get(key, check)
                if entry["error"]:
                    continue
                for fact, answer in zip(task.facts, entry["answers"]):
                    results.append((label, fact.kind, ANSWER_SCORES.get(answer, 0.0)))

    labels = sorted({label for label, _, _ in results})
    rows = []
    for kind in list(KIND_NAMES) + ["all"]:
        row = [KIND_NAMES.get(kind, "все указания")]
        for label in labels:
            values = [score for lbl, k, score in results if lbl == label and (kind == "all" or k == kind)]
            row.append(f"{100 * statistics.mean(values):.0f}% (n={len(values)})" if values else "—")
        rows.append(row)
    return ("## Управляемость генераторов\n\n"
            f"{len(TASKS)} сцен с проверяемыми фактами, по {repeats} картины. В ячейке — доля "
            "выполненных указаний.\n\n"
            + markdown_table(["Тип указания"] + labels, rows) + "\n")
