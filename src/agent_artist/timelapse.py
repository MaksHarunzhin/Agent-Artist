"""
Таймлапс эволюции картины: как художники шаг за шагом приходили к результату.

Из log.json прогона собираются:
- timelapse.webp и timelapse.gif — художники рядом, итерация за итерацией с плавными
  переходами; на каждой панели картина, оценка эксперта и его главное замечание.
  WebP легче и чище (для сайта), GIF вставляется в презентации;
- evolution.png — статичная сетка «итерации × художники» для отчёта и сайта.
"""

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .runlogs import has_scores, read_json

PANEL_SIZE = 512            # сторона картины на панели
PADDING = 16
HEADER_HEIGHT = 64          # название идеи над панелями
CAPTION_HEIGHT = 150        # оценка и замечание эксперта под картиной
LABEL_HEIGHT = 34           # подпись художника над картиной

HOLD_MS = 2600              # сколько показывается каждая итерация
FINAL_HOLD_MS = 5000        # последний кадр — дольше
TRANSITION_MS = 70
# Промежуточных кадров плавного перехода. В GIF их меньше: каждый кадр там тяжёлый
WEBP_TRANSITION_STEPS = 6
GIF_TRANSITION_STEPS = 3
WEBP_QUALITY = 80

BACKGROUND = (24, 24, 28)
PANEL_BACKGROUND = (40, 40, 46)
TEXT = (235, 235, 240)
MUTED_TEXT = (160, 160, 170)

# Шрифты с кириллицей: macOS, Linux, Windows
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

GENERATOR_NAMES = {"local": "SDXL-Turbo", "yandex_art": "Yandex ART"}


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def artist_caption(artist: dict) -> str:
    """Подпись художника в три строки: имя, генератор, язык промпта."""
    generator = GENERATOR_NAMES.get(artist["generator"], artist["generator"])
    return (f"Художник {artist['artist_id']}\n{generator}\n"
            f"промпт на {artist.get('prompt_language', '')}")


def text_column_width(texts: list[str], font) -> int:
    """Ширина колонки подписей — по самой длинной подписи, с отступами."""
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    return 2 * PADDING + max(
        measure.multiline_textbbox((0, 0), text, font=font, spacing=6)[2] for text in texts)


def score_color(score: int) -> tuple[int, int, int]:
    """От красного (плохо) через жёлтый к зелёному (хорошо)."""
    if score >= 8:
        return (90, 190, 110)
    if score >= 5:
        return (225, 185, 70)
    return (220, 90, 80)


class TimelapseRenderer:
    def __init__(self, log: dict, run_dir: Path):
        self.log = log
        self.run_dir = run_dir
        self.artists = log["artists"]
        self.history = log["history"]
        self.iterations = max(len(records) for records in self.history.values())

        self.title_font = load_font(26)
        self.label_font = load_font(20)
        self.score_font = load_font(22)
        self.text_font = load_font(16)

        self.width = PADDING + len(self.artists) * (PANEL_SIZE + PADDING)
        self.height = HEADER_HEIGHT + LABEL_HEIGHT + PANEL_SIZE + CAPTION_HEIGHT + PADDING

    def artist_label(self, artist: dict) -> str:
        generator = GENERATOR_NAMES.get(artist["generator"], artist["generator"])
        language = artist.get("prompt_language", "")[:3]
        return f"Художник {artist['artist_id']} · {generator} · {language}."

    def load_picture(self, record: dict) -> Image.Image:
        """Картина итерации, а если генерации не было — серая заглушка с причиной."""
        if record.get("image_path"):
            image_path = self.run_dir / Path(record["image_path"]).name
            if image_path.exists():
                return Image.open(image_path).convert("RGB").resize((PANEL_SIZE, PANEL_SIZE))

        placeholder = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), PANEL_BACKGROUND)
        draw = ImageDraw.Draw(placeholder)
        reason = "Генератор отказал" if record.get("error") else "Нет картины"
        draw.text((PANEL_SIZE // 2, PANEL_SIZE // 2), reason, font=self.score_font,
                  fill=MUTED_TEXT, anchor="mm")
        return placeholder

    def draw_caption(self, draw: ImageDraw.ImageDraw, x: int, y: int, record: dict) -> None:
        """Итерация, оценка с полоской и главное замечание эксперта."""
        score = record["review"]["score"]
        draw.text((x, y), f"Итерация {record['iteration']}", font=self.label_font, fill=MUTED_TEXT)
        draw.text((x + PANEL_SIZE, y), f"{score}/10", font=self.score_font,
                  fill=score_color(score), anchor="ra")

        bar_y = y + 32
        draw.rounded_rectangle((x, bar_y, x + PANEL_SIZE, bar_y + 8), radius=4,
                               fill=PANEL_BACKGROUND)
        if score > 0:
            draw.rounded_rectangle((x, bar_y, x + PANEL_SIZE * score // 10, bar_y + 8),
                                   radius=4, fill=score_color(score))

        if record.get("error"):
            # Отказ генератора интереснее показать через промпт, который не пропустили
            remark = f"Генератор отклонил промпт: «{record['prompt']}»"
        else:
            problems = record["review"].get("problems") or []
            remark = "Эксперт: " + (str(problems[0]) if problems
                                    else record["review"].get("instructions", ""))
        all_lines = textwrap.wrap(remark, width=58)
        lines = all_lines[:4]
        if len(all_lines) > 4:
            lines[-1] = lines[-1].rstrip(" .,;«»") + "…"
        draw.multiline_text((x, bar_y + 20), "\n".join(lines), font=self.text_font,
                            fill=TEXT, spacing=5)

    def render_frame(self, iteration_index: int) -> Image.Image:
        """Кадр одной итерации: все художники рядом."""
        frame = Image.new("RGB", (self.width, self.height), BACKGROUND)
        draw = ImageDraw.Draw(frame)
        draw.text((PADDING, HEADER_HEIGHT // 2), self.log["idea"], font=self.title_font,
                  fill=TEXT, anchor="lm")
        mode = "общий контекст" if self.log.get("shared_questions") else "раздельный контекст"
        draw.text((self.width - PADDING, HEADER_HEIGHT // 2), mode, font=self.label_font,
                  fill=MUTED_TEXT, anchor="rm")

        for column, artist in enumerate(self.artists):
            x = PADDING + column * (PANEL_SIZE + PADDING)
            records = self.history.get(artist["artist_id"], [])
            draw.text((x, HEADER_HEIGHT + 4), self.artist_label(artist), font=self.label_font,
                      fill=TEXT)
            if not records:
                continue
            # Если художник сделал меньше итераций, показываем его последнюю картину
            record = records[min(iteration_index, len(records) - 1)]
            frame.paste(self.load_picture(record), (x, HEADER_HEIGHT + LABEL_HEIGHT))
            self.draw_caption(draw, x, HEADER_HEIGHT + LABEL_HEIGHT + PANEL_SIZE + 12, record)

        return frame

    @staticmethod
    def with_transitions(frames: list[Image.Image],
                         steps: int) -> tuple[list[Image.Image], list[int]]:
        """Добавить между итерациями кадры плавного перехода. Возвращает (кадры, длительности)."""
        sequence, durations = [], []
        for index, frame in enumerate(frames):
            sequence.append(frame)
            is_last = index == len(frames) - 1
            durations.append(FINAL_HOLD_MS if is_last else HOLD_MS)
            if not is_last:
                for step in range(1, steps + 1):
                    sequence.append(Image.blend(frame, frames[index + 1], step / (steps + 1)))
                    durations.append(TRANSITION_MS)
        return sequence, durations

    def save_timelapse(self, webp_path: Path, gif_path: Path) -> None:
        frames = [self.render_frame(index) for index in range(self.iterations)]

        sequence, durations = self.with_transitions(frames, WEBP_TRANSITION_STEPS)
        sequence[0].save(webp_path, save_all=True, append_images=sequence[1:],
                         duration=durations, loop=0, quality=WEBP_QUALITY)

        sequence, durations = self.with_transitions(frames, GIF_TRANSITION_STEPS)
        sequence[0].save(gif_path, save_all=True, append_images=sequence[1:],
                         duration=durations, loop=0)

    def save_evolution_sheet(self, path: Path) -> None:
        """Сетка: строки — художники, столбцы — итерации."""
        thumb = 256
        labels = [artist_caption(artist) for artist in self.artists]
        label_width = text_column_width(labels, self.label_font)

        sheet = Image.new("RGB", (label_width + self.iterations * (thumb + PADDING) + PADDING,
                                  HEADER_HEIGHT + len(self.artists) * (thumb + 40 + PADDING)),
                          BACKGROUND)
        draw = ImageDraw.Draw(sheet)
        draw.text((PADDING, HEADER_HEIGHT // 2), self.log["idea"], font=self.title_font,
                  fill=TEXT, anchor="lm")

        for row, artist in enumerate(self.artists):
            y = HEADER_HEIGHT + row * (thumb + 40 + PADDING)
            draw.multiline_text((PADDING, y + thumb // 2), labels[row],
                                font=self.label_font, fill=TEXT, anchor="lm", spacing=6)

            for column, record in enumerate(self.history.get(artist["artist_id"], [])):
                x = label_width + column * (thumb + PADDING)
                sheet.paste(self.load_picture(record).resize((thumb, thumb)), (x, y))
                score = record["review"]["score"]
                draw.text((x, y + thumb + 8), f"№{record['iteration']}", font=self.text_font,
                          fill=MUTED_TEXT)
                draw.text((x + thumb, y + thumb + 6), f"{score}/10", font=self.label_font,
                          fill=score_color(score), anchor="ra")

        sheet.save(path)


def render_run(run_dir: Path) -> list[Path] | None:
    """Собрать таймлапс и evolution.png для прогона. None — если в логе нет итераций."""
    log = read_json(run_dir / "log.json")
    if not has_scores(log):
        return None

    renderer = TimelapseRenderer(log, run_dir)
    paths = [run_dir / "timelapse.webp", run_dir / "timelapse.gif", run_dir / "evolution.png"]
    renderer.save_timelapse(paths[0], paths[1])
    renderer.save_evolution_sheet(paths[2])
    return paths


EXPERT_NAMES = {"cinematic": "кинематографист", "minimalist": "минималист",
                "baroque": "любитель барокко", "child": "ребёнок 6 лет",
                "peredvizhnik": "передвижник"}


def column_title(log: dict, label: str) -> str:
    """Подпись столбца галереи: эксперт, идея или режим прогона."""
    if label == "expert":
        name = log.get("expert", "cinematic")
        return EXPERT_NAMES.get(name, name)
    if label == "mode":
        return "общий контекст" if log.get("shared_questions") else "раздельный контекст"
    return log["idea"]


def render_gallery(log_paths: list[Path], output_path: Path, label: str, title: str) -> None:
    """Сетка лучших картин: строки — художники, столбцы — прогоны с подписью label."""
    logs = [(path.parent, read_json(path)) for path in log_paths]
    logs = [(run_dir, log) for run_dir, log in logs if has_scores(log)]
    artists = {artist["artist_id"]: artist for _, log in logs for artist in log["artists"]}

    thumb, caption = 320, 34
    title_font, label_font, text_font = load_font(26), load_font(19), load_font(16)
    labels = [artist_caption(artist) for artist in artists.values()]
    label_width = text_column_width(labels, label_font)
    header = 64 + 56
    width = label_width + len(logs) * (thumb + PADDING) + PADDING
    height = header + len(artists) * (thumb + caption + PADDING)

    gallery = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(gallery)
    draw.text((PADDING, 32), title, font=title_font, fill=TEXT, anchor="lm")

    for column, (run_dir, log) in enumerate(logs):
        x = label_width + column * (thumb + PADDING)
        title_lines = textwrap.wrap(column_title(log, label), width=28)[:2]
        draw.multiline_text((x + thumb // 2, header - 30), "\n".join(title_lines),
                            font=label_font, fill=TEXT, anchor="mm", align="center", spacing=4)
        for row, artist_id in enumerate(artists):
            y = header + row * (thumb + caption + PADDING)
            if column == 0:
                draw.multiline_text((PADDING, y + thumb // 2), labels[row], font=label_font,
                                    fill=TEXT, anchor="lm", spacing=6)
            records = log["history"].get(artist_id, [])
            drawn = [record for record in records if record.get("image_path")]
            if not drawn:
                if records:
                    draw.rectangle((x, y, x + thumb, y + thumb), fill=PANEL_BACKGROUND)
                    draw.text((x + thumb // 2, y + thumb // 2), "генератор отказал",
                              font=label_font, fill=MUTED_TEXT, anchor="mm")
                continue
            best = max(drawn, key=lambda record: record["review"]["score"])
            picture = Image.open(run_dir / Path(best["image_path"]).name).convert("RGB")
            gallery.paste(picture.resize((thumb, thumb)), (x, y))
            score = best["review"]["score"]
            draw.text((x, y + thumb + 8), f"итерация {best['iteration']}", font=text_font,
                      fill=MUTED_TEXT)
            draw.text((x + thumb, y + thumb + 6), f"{score}/10", font=label_font,
                      fill=score_color(score), anchor="ra")

    gallery.save(output_path)
