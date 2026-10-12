import os
os.environ["HF_HUB_OFFLINE"] = "1"   # ДО импорта diffusers, иначе не сработает

import base64
import json
import re
import time
from pathlib import Path

import httpx
import ollama
import torch
from diffusers import AutoPipelineForText2Image
from openai import OpenAI

# ---------------- Настройки ----------------
IDEA = "Золотистый ретривер"                 # идея картины

OLLAMA_HOST = "http://127.0.0.1:11434"
TEXT_MODEL = "qwen3:8b"              # художник и эксперт
VISION_MODEL = "qwen3-vl:8b"         # описание картинок
IMAGE_MODEL = "stabilityai/sdxl-turbo"

MIN_IMAGES = 5          # минимум картин
MAX_IMAGES = 8          # жёсткий предел
TARGET_SCORE = 8        # 0 = остановиться ровно на MIN_IMAGES
START_ROUNDS = 3        # раундов вопросов до первой картины
ITER_ROUNDS = 1         # раундов вопросов после каждой критики
MAX_QUESTIONS = 4       # вопросов за один раунд
DISABLE_THINKING = True # добавлять /no_think для моделей qwen3
OUT_DIR = Path("runs") / time.strftime("%Y%m%d_%H%M%S")

# Клиенты: Responses API (OpenAI-совместимый) и нативный Ollama.
# trust_env=False — игнорировать системный прокси для локальных запросов.
client = OpenAI(
    base_url=f"{OLLAMA_HOST}/v1/",
    api_key="ollama",   # обязателен для клиента, Ollama его игнорирует
    http_client=httpx.Client(trust_env=False, timeout=600),
)
native = ollama.Client(host=OLLAMA_HOST, trust_env=False)

# Вкусы эксперта: меняйте под себя, от них зависят все его решения
EXPERT_TASTE = """- Любишь кинематографичные кадры с выразительным светом и глубиной.
- Предпочитаешь живопись и иллюстрацию фотореализму.
- Ценишь, когда в картине есть история и напряжение момента.
- Не любишь перегруженные сцены: один ясный главный сюжет, продуманный фон.
- Предпочитаешь насыщенную, но гармоничную палитру."""

# ---------------- Промпты ролей ----------------
ASK_SYSTEM = """Ты художник-иллюстратор. Тебе заказали картину, но все решения о деталях
принимает арт-эксперт (заказчик). Тебе дают идею, уже согласованные с экспертом детали
и, возможно, его критику последней версии картины.
Реши, хватает ли информации, чтобы написать МАКСИМАЛЬНО точный и подробный промпт.
Если не хватает — задай эксперту конкретные вопросы о том, что ещё не согласовано:
сюжет, персонажи и их количество, внешность, одежда, позы, эмоции, место, время суток,
погода, стиль, палитра, освещение, ракурс, композиция, детали фона.
Если есть критика — уточни то, что в ней неясно или требует выбора.
Не повторяй вопросы, на которые уже есть ответ. Не больше 4 вопросов.
Ответь ТОЛЬКО JSON-объектом, без пояснений и markdown:
{"ready": false, "questions": ["...", "..."]}
Если информации достаточно: {"ready": true, "questions": []}"""

EXPERT_ANSWER_SYSTEM = f"""Ты арт-эксперт и заказчик картины. Твои вкусы:
{EXPERT_TASTE}
Художник задаёт тебе уточняющие вопросы. Отвечай на каждый конкретно и решительно,
исходя из своих вкусов и идеи. Никогда не отвечай «на ваше усмотрение» — принимай решение сам.
Не противоречь своим прежним ответам из диалога.
Ответь ТОЛЬКО JSON-объектом, без пояснений и markdown, ответы в порядке вопросов:
{{"answers": ["ответ на вопрос 1", "ответ на вопрос 2"]}}"""

ARTIST_SYSTEM = """Ты художник-иллюстратор и эксперт по промптам для Stable Diffusion.
Тебе дают идею, все согласованные с экспертом детали и, возможно, его критику прошлой версии.
Учти КАЖДУЮ согласованную деталь. Если есть критика — исправь каждое замечание.
Напиши промпт НА АНГЛИЙСКОМ: не длиннее 60 слов, фразы через запятую,
самое важное (сюжет, персонажи, стиль) в начале.
Ответь ТОЛЬКО JSON-объектом, без пояснений и markdown:
{"reasoning": "по-русски: что учёл и что изменил",
 "prompt": "...", "negative_prompt": "..."}"""

VLM_PROMPT = """Опиши изображение объективно и подробно: какие объекты есть,
сколько их, где они расположены, цвета, стиль, освещение, фон, любой текст.
Отдельно перечисли дефекты: искажения анатомии, лишние или слитые предметы,
размытость, артефакты. Не додумывай того, чего на изображении нет."""

EXPERT_REVIEW_SYSTEM = f"""Ты строгий арт-эксперт и заказчик картины. Твои вкусы:
{EXPERT_TASTE}
Тебе дают идею, детали, которые ты сам согласовал с художником, его промпт
и объективное описание получившейся картины. Картину ты не видишь — опирайся
только на описание. Сравни задуманное с полученным.
Оцени соответствие согласованным деталям и качество по шкале 1–10.
Дай художнику конкретные инструкции, что изменить в следующей версии.
Ответь ТОЛЬКО JSON-объектом, без пояснений и markdown:
{{"score": 7, "matches": ["..."], "problems": ["..."],
 "instructions_for_artist": "..."}}"""


# ---------------- Вызовы через Responses API ----------------
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_text(text: str) -> str:
    """Убрать блок рассуждений <think>, если модель его всё же выдала."""
    return THINK_RE.sub("", text).strip()


def extract_json(text: str) -> dict:
    """Достать JSON-объект из ответа, даже если вокруг есть markdown или текст."""
    text = clean_text(text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise json.JSONDecodeError("JSON-объект не найден", text, 0)
    return json.loads(text[start:end + 1])


def chat_json(model: str, system: str, user: str, retries: int = 3) -> dict:
    """Запрос к LLM через Responses API с JSON-ответом и повтором при ошибке."""
    if DISABLE_THINKING:
        user = f"{user}\n\n/no_think"
    for attempt in range(1, retries + 1):
        resp = client.responses.create(
            model=model,
            instructions=system,   # системный промпт
            input=user,
            temperature=0.7,
        )
        try:
            return extract_json(resp.output_text)
        except json.JSONDecodeError:
            print(f"  ! невалидный JSON (попытка {attempt}), повторяю")
    raise RuntimeError(f"{model} не вернула валидный JSON")


_vlm_native_fallback = False


def describe(image_path: Path) -> str:
    """Описание картинки: сначала через Responses API, при ошибке — нативно."""
    global _vlm_native_fallback
    prompt = f"{VLM_PROMPT}\n\n/no_think" if DISABLE_THINKING else VLM_PROMPT

    if not _vlm_native_fallback:
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        try:
            resp = client.responses.create(
                model=VISION_MODEL,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image",
                         "image_url": f"data:image/png;base64,{b64}"},
                    ],
                }],
                temperature=0.2,
            )
            text = clean_text(resp.output_text)
            if text:
                unload(VISION_MODEL)
                return text
            raise ValueError("пустой ответ")
        except Exception as e:
            print(f"  ! Responses API не принял изображение ({e}); "
                  f"дальше VLM работает через нативный API Ollama")
            _vlm_native_fallback = True

    resp = native.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": VLM_PROMPT,
                   "images": [str(image_path)]}],
        think=False,
        keep_alive=0,
        options={"temperature": 0.2},
    )
    return clean_text(resp.message.content)


def unload(model: str):
    """Выгрузить модель из видеопамяти (в Responses API такой функции нет)."""
    try:
        native.generate(model=model, prompt="", keep_alive=0)
    except Exception:
        pass


# ---------------- Общий контекст ----------------
def format_dialogue(dialogue: list[dict], limit: int = 40) -> str:
    if not dialogue:
        return "пока ничего не согласовано"
    return "\n".join(f"Художник: {d['q']}\nЭксперт: {d['a']}" for d in dialogue[-limit:])


def build_context(history: list[dict], dialogue: list[dict]) -> str:
    ctx = (f"Идея картины: {IDEA}\n\n"
           f"Согласованные детали (вопросы художника и ответы эксперта):\n"
           f"{format_dialogue(dialogue)}\n")
    if history:
        last = history[-1]
        ctx += (f"\nПрошлый промпт: {last['prompt']}"
                f"\nОценка эксперта: {last['score']}/10"
                f"\nПроблемы: {'; '.join(map(str, last['problems']))}"
                f"\nИнструкции эксперта: {last['instructions']}\n")
    return ctx


# ---------------- Роли ----------------
def consult(history: list[dict], dialogue: list[dict], rounds: int, min_rounds: int = 0):
    """Художник задаёт вопросы эксперту, эксперт отвечает по своим вкусам."""
    for r in range(1, rounds + 1):
        ask = chat_json(TEXT_MODEL, ASK_SYSTEM, build_context(history, dialogue))
        questions = [str(q) for q in ask.get("questions", []) if q][:MAX_QUESTIONS]
        if (ask.get("ready") and r > min_rounds) or not questions:
            print("[Художник] Информации достаточно.")
            return

        q_text = "\n".join(f"{k}. {q}" for k, q in enumerate(questions, 1))
        reply = chat_json(TEXT_MODEL, EXPERT_ANSWER_SYSTEM,
                          build_context(history, dialogue)
                          + f"\nНовые вопросы художника:\n{q_text}")
        answers = reply.get("answers", [])

        print(f"--- Раунд вопросов {r} ---")
        for q, a in zip(questions, answers):
            dialogue.append({"q": q, "a": str(a)})
            print(f"[Художник] {q}\n[Эксперт]  {a}")


def artist(history: list[dict], dialogue: list[dict]) -> dict:
    return chat_json(TEXT_MODEL, ARTIST_SYSTEM, build_context(history, dialogue))


def expert_review(prompt: str, description: str,
                  history: list[dict], dialogue: list[dict]) -> dict:
    scores = ", ".join(str(h["score"]) for h in history) or "нет"
    msg = (f"Идея картины: {IDEA}\n\n"
           f"Согласованные детали:\n{format_dialogue(dialogue)}\n\n"
           f"Промпт художника:\n{prompt}\n\n"
           f"Описание получившейся картины:\n{description}\n\n"
           f"Оценки прошлых версий: {scores}")
    return chat_json(TEXT_MODEL, EXPERT_REVIEW_SYSTEM, msg)


# ---------------- Генерация изображений ----------------
class ImageGenerator:
    def __init__(self):
        if torch.cuda.is_available():
            self.device, dtype = "cuda", torch.float16
        elif torch.backends.mps.is_available():
            self.device, dtype = "mps", torch.float16
        else:
            self.device, dtype = "cpu", torch.float32
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            IMAGE_MODEL, dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()  # экономия VRAM
        else:
            self.pipe.to(self.device)

    def generate(self, prompt: str, negative: str, path: Path, seed: int):
        g = torch.Generator("cpu").manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative or None,  # SDXL-Turbo его игнорирует
            num_inference_steps=4,
            guidance_scale=0.0,     # для SDXL-Turbo; для SDXL base ставьте ~7
            width=512, height=512,
            generator=g,
        ).images[0]
        image.save(path)


# ---------------- Оркестратор ----------------
def save_log(history: list[dict], dialogue: list[dict]):
    (OUT_DIR / "log.json").write_text(
        json.dumps({"idea": IDEA, "expert_taste": EXPERT_TASTE,
                    "dialogue": dialogue, "history": history},
                   ensure_ascii=False, indent=2), encoding="utf-8")


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    dialogue: list[dict] = []

    print(f"Идея: {IDEA}\n\n=== Согласование деталей ===")
    consult(history, dialogue, rounds=START_ROUNDS, min_rounds=1)
    save_log(history, dialogue)

    generator = ImageGenerator()
    i = 0
    while True:
        i += 1
        print(f"\n=== Итерация {i} ===")

        if history:   # после критики художник может уточнить детали
            consult(history, dialogue, rounds=ITER_ROUNDS)

        art = artist(history, dialogue)
        print(f"[Художник] {art.get('reasoning', '')}\n  prompt: {art['prompt']}")

        unload(TEXT_MODEL)   # освободить VRAM под генерацию
        img_path = OUT_DIR / f"image_{i:02d}.png"
        seed = 1000 + i
        generator.generate(art["prompt"], art.get("negative_prompt", ""), img_path, seed)
        print(f"[Генератор] {img_path}")

        description = describe(img_path)
        print(f"[VLM] {description[:300]}...")

        review = expert_review(art["prompt"], description, history, dialogue)
        score = int(review.get("score", 0))
        print(f"[Эксперт] {score}/10: {review.get('instructions_for_artist', '')}")

        history.append({
            "iteration": i, "seed": seed, "image": str(img_path),
            "prompt": art["prompt"],
            "negative_prompt": art.get("negative_prompt", ""),
            "artist_reasoning": art.get("reasoning", ""),
            "description": description, "score": score,
            "problems": review.get("problems", []),
            "instructions": review.get("instructions_for_artist", ""),
        })
        save_log(history, dialogue)

        if i >= MIN_IMAGES and (score >= TARGET_SCORE or i >= MAX_IMAGES):
            break

    best = max(history, key=lambda h: h["score"])
    print(f"\nГотово: {i} картин. Лучшая — {best['image']} ({best['score']}/10)")


if __name__ == "__main__":
    run()