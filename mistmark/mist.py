"""Measure actual photographed stripe contrast and highlights on misted glass."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from image_lab.validation import integer

from .photo import decode, pixels, png, MAX_IMAGE_BYTES


LUMA = np.array([0.2126, 0.7152, 0.0722])


def settings(value, size):
    width, height = size
    left = integer(value.get("left"), "左界 X", 0, width - 1)
    top = integer(value.get("top"), "上界 Y", 0, height - 1)
    return {
        "left": left, "right": integer(value.get("right"), "右界 X", left, width - 1),
        "top": top, "bottom": integer(value.get("bottom"), "下界 Y", top, height - 1),
        "focus": integer(value.get("focus"), "截线 Y", 0, height - 1),
        "cutoff": integer(value.get("cutoff"), "亮点下限", 1, 254),
    }


def defaults(size):
    width, height = size
    return settings({
        "left": width // 7, "right": width * 6 // 7, "top": height // 5,
        "bottom": height * 4 // 5, "focus": height // 2, "cutoff": 175,
    }, size)


def contrast(low, high):
    levels = high + low
    return np.divide(high - low, levels, out=np.zeros_like(levels), where=levels > 0)


def point(image, x, y, config):
    rgb = image[y, x]
    luma = float(rgb @ LUMA)
    inside = config["left"] <= x <= config["right"] and config["top"] <= y <= config["bottom"]
    return {"x": x, "y": y, "rgb": rgb.tolist(), "brightness": luma,
            "highlight": luma >= config["cutoff"] if inside else None}


def profile(image, config):
    y = config["focus"]
    if not config["top"] <= y <= config["bottom"]:
        return []
    return [point(image, x, y, config) for x in range(config["left"], config["right"] + 1)]


def row_metrics(image, config):
    left, right, top, bottom = (config[key] for key in ("left", "right", "top", "bottom"))
    values = image[top:bottom + 1, left:right + 1] @ LUMA
    lows, highs = values.min(axis=1), values.max(axis=1)
    high_points = np.count_nonzero(values >= config["cutoff"], axis=1)
    ratios = contrast(lows, highs)
    row_results = [
        {"y": y, "low": float(lo), "high": float(hi), "contrast": float(ratio), "highlights": int(count)}
        for y, lo, hi, ratio, count in zip(range(top, bottom + 1), lows, highs, ratios, high_points)
    ]
    return row_results, values.shape[1]


def report(image, config):
    rows = profile(image, config)
    metrics, width = row_metrics(image, config)
    reading = None
    if rows:
        reading = {**metrics[config["focus"] - config["top"]],
                   "dark_x": min(rows, key=lambda p:p["brightness"])["x"],
                   "light_x": max(rows, key=lambda p:p["brightness"])["x"]}
    area = width * len(metrics)
    marked = sum(row["highlights"] for row in metrics)
    return {"reading": reading, "profile": rows, "rows": metrics, "focus": config["focus"],
            "area": {"pixels": area, "highlight_pixels": marked, "fraction": marked / area}}


def practice():
    width, height = 780, 466
    cfg = settings({"left": 138, "right": 640, "top": 119, "bottom": 352, "focus": 154,
                    "cutoff": 178}, (width, height))
    y, x = np.indices((height,width))
    glass = (82 <= x) & (x <= 697) & (72 <= y) & (y <= 404)
    bars = glass & (115 <= x) & (x <= 661) & (101 <= y) & (y <= 371)
    polarity = np.where((x - 116) // 33 % 2 == 0, 1, -1)
    fog = np.exp(-((y - 276) / 41) ** 2)
    sun = 4 * np.cos((x - 420) / 138)
    texture = 2.0 * np.sin(x * 0.24 + y * 0.65) + 1.6 * np.sin(y * 0.64 - x * 0.77)
    chart = 130 + 13 * fog + polarity * (45 - 34 * fog) + sun
    beads = np.zeros_like(fog)
    for bx, by, radius, intensity in ((268,244,7,28),(412,281,6,18),(541,261,11,33),
                                       (598,296,4,29),(335,307,9,25),(201,285,5,18)):
        distance = np.hypot(x - bx, y - by)
        beads += intensity * np.exp(-((distance - radius) / 1.6)**2)
        beads -= 9 * np.exp(-(distance / max(2, radius-2))**2)
    base = np.where(bars,chart, np.where(glass,83,43)) + beads + texture
    photo = Image.fromarray((base[...,None] + np.array([-13,2,21])).clip(0,255).astype("uint8"),"RGB")
    draw = ImageDraw.Draw(photo)
    draw.rounded_rectangle((82,72,697,404),radius=12,outline="#5ba4ae",width=3)
    draw.text((108,39),"DAMP GLASS / CONTRAST CARD",fill="#d4e6e9",font=ImageFont.load_default(size=13))
    draw.text((108,428),"SIMULATED GLASS PHOTO",fill="#bae3e4",font=ImageFont.load_default(size=12))
    return photo.size, png(photo), cfg
