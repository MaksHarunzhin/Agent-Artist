"""
Временные картинки для сайта: вырезает отдельные картины из готовых сеток в docs/media
(«испорченный телефон», «пять заказчиков», «культурная асимметрия») и кладёт их
в site/preview/ под теми же путями, что и настоящие картинки в site/img/.

Сайт сначала ищет картинку в site/img/, а если её там нет — в site/preview/.
Когда в репозиторий загрузят site/img/ (site/export_images.py), вырезки станут не нужны.

Запуск из корня проекта:  python site/make_previews.py
"""
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "docs" / "media"
RUNS = ROOT / "runs"
OUT = ROOT / "site" / "preview"
PADDING = 16  # как в src/agent_artist/timelapse.py


def crop_grid(sheet: Path, thumb: int, caption: int, header: int, rows: int, columns: int):
    """Возвращает функцию cell(row, col) → картинка, с учётом того, что сетку уменьшили."""
    image = Image.open(sheet).convert("RGB")
    original_height = header + rows * (thumb + caption + PADDING)
    scale = image.height / original_height
    original_width = image.width / scale
    label_width = original_width - columns * (thumb + PADDING) - PADDING

    def cell(row: int, col: int) -> Image.Image:
        x = (label_width + col * (thumb + PADDING)) * scale
        y = (header + row * (thumb + caption + PADDING)) * scale
        size = thumb * scale
        inset = 1.5  # отступ, чтобы не захватить фон по краям
        return image.crop((round(x + inset), round(y + inset), round(x + size - inset), round(y + size - inset)))

    return cell


def save(picture: Image.Image, rel: str) -> None:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    picture.save(path, "JPEG", quality=88)


def best_record(records):
    drawn = [r for r in records if r.get("image_path")]
    return max(drawn, key=lambda r: r["review"]["score"]) if drawn else None


def main() -> None:
    count = 0

    # «Испорченный телефон»: строки A, B, C × шаги 0–5
    for number, name in [("01", "telephone_lighthouse"), ("02", "telephone_baba_yaga"), ("03", "telephone_bear_skin")]:
        cell = crop_grid(MEDIA / f"{name}.jpg", thumb=256, caption=74, header=64, rows=3, columns=6)
        for row, artist in enumerate("ABC"):
            chain = json.loads((RUNS / "telephone" / number / artist / "chain.json").read_text(encoding="utf-8"))
            for step in chain["steps"]:
                if step["image"]:
                    save(cell(row, step["step"]), f"telephone/{number}/{artist}/{Path(step['image']).stem}.jpg")
                    count += 1

    # «Пять заказчиков»: строки A, B × столбцы-заказчики, лучшая картина каждого
    cell = crop_grid(MEDIA / "clients_gallery.jpg", thumb=320, caption=34, header=120, rows=2, columns=5)
    for col, client in enumerate(["baroque", "child", "cinematic", "minimalist", "peredvizhnik"]):
        log = json.loads((RUNS / "clients" / client / "log.json").read_text(encoding="utf-8"))
        for row, artist in enumerate("AB"):
            best = best_record(log["history"][artist])
            if best:
                save(cell(row, col), f"clients/{client}/{Path(best['image_path']).stem}.jpg")
                count += 1

    # «Культурная асимметрия»: строки A, B, C × пять идей. В сетке — один прогон из двух;
    # выбираем тот, у которого лучшие картины совпадают с подписями сетки.
    expected = {  # (итерация, оценка) лучших картин по сетке, None — генератор отказал
        "01": {"A": (1, 4), "B": (2, 6), "C": (1, 6)},
        "02": {"A": (1, 3), "B": (2, 6), "C": (2, 7)},
        "03": {"A": (1, 3), "B": (1, 4), "C": None},
        "04": {"A": (1, 4), "B": (1, 4), "C": (2, 8)},
        "05": {"A": (2, 4), "B": (2, 8), "C": (1, 7)},
    }
    cell = crop_grid(MEDIA / "culture_gallery.jpg", thumb=320, caption=34, header=120, rows=3, columns=5)
    for col, idea in enumerate(expected):
        for run in ("01_isolated", "02_isolated"):
            log = json.loads((RUNS / "culture" / idea / run / "log.json").read_text(encoding="utf-8"))
            bests = {a: best_record(log["history"][a]) for a in "ABC"}
            got = {a: (int(b["iteration"]), b["review"]["score"]) if b else None for a, b in bests.items()}
            if got == expected[idea]:
                for row, artist in enumerate("ABC"):
                    if bests[artist]:
                        save(cell(row, col), f"culture/{idea}/{run}/{Path(bests[artist]['image_path']).stem}.jpg")
                        count += 1
                print(f"  culture/{idea} — из прогона {run}")
                break
        else:
            print(f"  ! culture/{idea}: не нашёл прогон, совпадающий с сеткой")

    # Сетки эволюции: строки A, B × итерации
    for name, rel, iterations in [("evolution_lighthouse_shared", "experiment_lighthouse/01_shared", 3),
                                  ("evolution_retriever", "20261013_221703", 8)]:
        cell = crop_grid(MEDIA / f"{name}.jpg", thumb=256, caption=40, header=64, rows=2, columns=iterations)
        log = json.loads((RUNS / rel / "log.json").read_text(encoding="utf-8"))
        for row, artist in enumerate("AB"):
            for col, record in enumerate(log["history"][artist]):
                if record.get("image_path"):
                    stem = Path(record["image_path"].replace("\\", "/")).stem
                    save(cell(row, col), f"{rel}/{stem}.jpg")
                    count += 1

    print(f"Готово: {count} вырезок в site/preview/")


if __name__ == "__main__":
    main()
