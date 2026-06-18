from PIL import Image, ImageDraw, ImageFont
import string
import sys

def stitch_figures_vertically(image_paths, output_path="output.png",
                               label_size=60, padding=20):
    """
    Stitch N figures vertically with panel labels (A, B, C, ...).

    Args:
        image_paths: list of paths to input images
        output_path: path for the output image
        label_size: font size for panel labels
        padding: padding around labels in pixels
    """
    images = [Image.open(p).convert("RGB") for p in image_paths]

    # Canvas dimensions
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)

    canvas = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Try to load a .ttf font, fallback to default with size support
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", label_size)
    except IOError:
        try:
            font = ImageFont.load_default(size=label_size)  # Pillow >= 10.1.0
        except TypeError:
            print("⚠️  Pillow version too old for resizable default font.")
            print("   Run: pip install --upgrade Pillow")
            font = ImageFont.load_default()

    # Labels: A, B, C, ... then AA, AB, etc. if N > 26
    labels = list(string.ascii_uppercase)

    y_offset = 0
    for i, img in enumerate(images):
        canvas.paste(img, (0, y_offset))

        # Draw panel label
        label = labels[i] if i < 26 else f"A{labels[i - 26]}"
        draw.text(
            (padding, y_offset + padding),
            label,
            fill=(0, 0, 0),   # white text
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0)   # black outline for visibility
        )

        y_offset += img.height

    canvas.save(output_path)
    print(f"✅ Saved stitched image to: {output_path}")


