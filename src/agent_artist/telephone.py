"""
«Испорченный телефон»: один агент объясняет картину другому, тот рисует по объяснению.

Шаг 0 — художник рисует по идее. Дальше слепой зритель описывает последнюю картину,
а следующий художник рисует только по этому описанию — саму картину он не видит.
Судья на каждом шаге оценивает, насколько картина ещё передаёт исходную идею.
Это асимметрия контекста в чистом виде: каждый знает только то, что ему пересказали.
"""

import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .config import ArtistConfig, Settings
from .generators import ImageGenerator, create_image_generators
from .llm import YandexLLM, create_yandex_client
from .prompts import BLIND_VIEWER_SYSTEM_PROMPT, TELEPHONE_JUDGE_SYSTEM_PROMPT
from .runlogs import read_json, write_json
from .studio import draw_with_rephrasing, write_fitting_prompt
from .timelapse import (BACKGROUND, MUTED_TEXT, PADDING, TEXT, artist_caption, load_font,
                        score_color, text_column_width)


@dataclass
class TelephoneStep:
    step: int
    told: str               # что художник получил: идея или пересказ зрителя
    prompt: str
    image: str | None
    seen_title: str         # как зритель назвал картину
    seen_description: str   # что зритель передал следующему художнику
    similarity: int         # насколько картина ещё передаёт исходную идею, 0–10
    lost: str
    error: str | None = None


class TelephoneChain:
    def __init__(self, settings: Settings, artist: ArtistConfig, generator: ImageGenerator,
                 client, output_dir: Path):
        folder_id = settings.yandex_folder_id
        self.settings = settings
        self.artist = artist
        self.generator = generator
        self.output_dir = output_dir
        self.artist_llm = YandexLLM(client, folder_id, settings.artist_llm_name)
        self.viewer = YandexLLM(client, folder_id, settings.expert_vision_llm_name)
        self.judge = YandexLLM(client, folder_id, settings.expert_llm_name)

    def run(self, steps: int) -> list[TelephoneStep]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        idea = self.settings.idea
        told = f"Идея картины: {idea}"
        chain = []

        for step in range(steps):
            print(f"[Телефон {self.artist.artist_id}] шаг {step}: {told[:120]}")
            prompt, _ = write_fitting_prompt(self.artist_llm, self.artist, self.generator, told)
            image_path = self.output_dir / f"step_{step:02d}.png"
            prompt, error = draw_with_rephrasing(self.artist_llm, self.artist, self.generator,
                                                 told, prompt, image_path,
                                                 self.settings.rephrase_attempts)
            if error is not None:
                # Без картины пересказывать нечего — цепочка обрывается
                print(f"  ! Генерация не удалась: {error}")
                chain.append(TelephoneStep(step, told, prompt, None, "", "", 0, "", error))
                break

            seen = self.viewer.ask_json(
                BLIND_VIEWER_SYSTEM_PROMPT, "Опиши эту картину.", image_path=image_path,
                required_fields=("title", "description"), temperature=0.2)
            verdict = self.judge.ask_json(
                TELEPHONE_JUDGE_SYSTEM_PROMPT,
                f"Исходная идея: {idea}\n\nКартина через {step} пересказов. "
                f"Зритель назвал её: {seen['title']}\nОписание: {seen['description']}",
                required_fields=("similarity",), temperature=0.2)

            chain.append(TelephoneStep(
                step=step, told=told, prompt=prompt, image=image_path.name,
                seen_title=str(seen["title"]), seen_description=str(seen["description"]),
                similarity=int(verdict["similarity"]), lost=str(verdict.get("lost", "")),
            ))
            print(f"  зритель: «{seen['title']}», сходство с идеей {verdict['similarity']}/10")

            # Следующий художник знает только пересказ — ни идеи, ни картины, ни названия:
            # название почти повторяет идею и передало бы её дальше в обход картины
            told = ("Нарисуй картину по рассказу человека, который её видел:\n"
                    f"{seen['description']}")

        write_json(self.output_dir / "chain.json",
                   {"idea": idea, "artist": asdict(self.artist),
                    "steps": [asdict(step) for step in chain]})
        return chain


def render_strip(chains: dict[str, dict], path: Path) -> None:
    """Лента: строка на художника, по шагам — картина, как её назвал зритель и сходство."""
    thumb = 256
    caption = 74
    label_font, text_font, title_font = load_font(18), load_font(15), load_font(26)
    steps = max(len(chain["steps"]) for chain in chains.values())
    labels = [artist_caption(chain["artist"]) for chain in chains.values()]
    label_width = text_column_width(labels, label_font)
    width = label_width + steps * (thumb + PADDING) + PADDING
    header = 64
    height = header + len(chains) * (thumb + caption + PADDING)

    strip = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(strip)
    idea = next(iter(chains.values()))["idea"]
    draw.text((PADDING, header // 2), f"Испорченный телефон: «{idea}»", font=title_font,
              fill=TEXT, anchor="lm")

    for row, (run_dir, chain) in enumerate(chains.items()):
        y = header + row * (thumb + caption + PADDING)
        draw.multiline_text((PADDING, y + thumb // 2), labels[row],
                            font=label_font, fill=TEXT, anchor="lm", spacing=6)

        for column, step in enumerate(chain["steps"]):
            x = label_width + column * (thumb + PADDING)
            if step["image"]:
                picture = Image.open(Path(run_dir) / step["image"]).convert("RGB")
                strip.paste(picture.resize((thumb, thumb)), (x, y))
                if column:
                    # Стрелка «пересказ» между соседними картинами
                    draw.text((x - PADDING // 2, y + thumb // 2), "›", font=title_font,
                              fill=MUTED_TEXT, anchor="mm")
            title = step["seen_title"] or "генератор отказал"
            draw.multiline_text((x, y + thumb + 6), _wrap(f"«{title}»", 30, 2),
                                font=text_font, fill=TEXT, spacing=3)
            draw.text((x + thumb, y + thumb + caption - 18), f"{step['similarity']}/10",
                      font=label_font, fill=score_color(step["similarity"]), anchor="ra")

    strip.save(path)


def _wrap(text: str, width: int, max_lines: int) -> str:
    lines = textwrap.wrap(text, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .,;«»") + "…»"
    return "\n".join(lines)


def run_telephone(settings: Settings, steps: int, root: Path) -> Path:
    """Цепочка на каждого художника из настроек и общая лента telephone.png."""
    client = create_yandex_client(settings.yandex_api_key, settings.yandex_folder_id)
    generators = create_image_generators(settings, client)

    chains = {}
    for artist in settings.artists:
        run_dir = root / artist.artist_id
        TelephoneChain(settings, artist, generators[artist.artist_id], client, run_dir).run(steps)
        chains[str(run_dir)] = read_json(run_dir / "chain.json")

    strip_path = root / "telephone.png"
    render_strip(chains, strip_path)
    return strip_path
