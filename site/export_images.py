"""
Готовит картинки для сайта: берёт все картины из runs/ и сохраняет их
лёгкими JPEG в site/img/ с той же структурой папок.

Оригиналы в runs/ не трогаются (они в .gitignore и весят сотни мегабайт),
а site/img/ можно спокойно загружать в репозиторий: ~25 МБ на все прогоны.

Запуск из корня проекта:
    python site/export_images.py
"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "site" / "img"
SIZE = 640       # длинная сторона, пикселей
QUALITY = 82     # качество JPEG


def main() -> None:
    sources = sorted(p for p in RUNS.rglob("*.png"))
    if not sources:
        print("В runs/ нет картинок .png — нечего экспортировать.")
        return

    done = skipped = 0
    for src in sources:
        dst = OUT / src.relative_to(RUNS).with_suffix(".jpg")
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(src).convert("RGB")
        image.thumbnail((SIZE, SIZE), Image.LANCZOS)
        image.save(dst, "JPEG", quality=QUALITY, optimize=True, progressive=True)
        done += 1

    total = sum(f.stat().st_size for f in OUT.rglob("*.jpg")) / 1e6
    print(f"Готово: {done} новых, {skipped} без изменений. site/img/ весит {total:.1f} МБ.")


if __name__ == "__main__":
    main()
