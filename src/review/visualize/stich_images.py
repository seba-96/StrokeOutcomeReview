from PIL import Image, ImageDraw, ImageFont
import string
from pathlib import Path

def _panel_label(i: int) -> str:
    # A, B, C, ... Z, AA, AB, ...
    label = ""
    n = i
    while True:
        n, rem = divmod(n, 26)
        label = string.ascii_uppercase[rem] + label
        if n == 0:
            break
        n -= 1
    return label

def stitch_figures_vertically(
    image_paths,
    output_path="output.png",
    label_size=60,
    padding=20,
    panels_output_dir=None,
    panel_prefix="forest_panel_",
):
    """
    Stitch N figures vertically with panel labels (A, B, C, ...),
    and optionally save each labeled panel separately.
    """
    images = [Image.open(p).convert("RGB") for p in image_paths]

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", label_size)
    except IOError:
        try:
            font = ImageFont.load_default(size=label_size)
        except TypeError:
            print("⚠️  Pillow version too old for resizable default font.")
            print("   Run: pip install --upgrade Pillow")
            font = ImageFont.load_default()

    # Save individual labeled panels if requested
    if panels_output_dir is not None:
        panels_output_dir = Path(panels_output_dir)
        panels_output_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(images):
            label = _panel_label(i)
            panel = img.copy()
            draw = ImageDraw.Draw(panel)
            draw.text(
                (padding, padding),
                label,
                fill=(0, 0, 0),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
            panel_path = panels_output_dir / f"{panel_prefix}{label}.png"
            panel.save(panel_path)
            print(f"✅ Saved panel {label} to: {panel_path}")

    # Stitch combined figure
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    canvas = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y_offset = 0
    for i, img in enumerate(images):
        canvas.paste(img, (0, y_offset))
        label = _panel_label(i)
        draw.text(
            (padding, y_offset + padding),
            label,
            fill=(0, 0, 0),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        y_offset += img.height

    canvas.save(output_path)
    print(f"✅ Saved stitched image to: {output_path}")




