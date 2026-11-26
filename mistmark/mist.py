"""Measure actual photographed stripe contrast and highlights on misted glass."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from image_lab.validation import integer

from .photo import decode, pixels, png, MAX_IMAGE_BYTES


LUMA = np.array([0.2126, 0.7152, 0.0722])
FOUR_WAY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
EIGHT_WAY = np.ones((3, 3), dtype=int)
HEART_MIN_PIXELS = 9


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


def droplets(image, config):
    """Dark-hearted droplets sealed by one solid bright edge inside the frame.

    A droplet qualifies when its dark heart (pixels genuinely below the
    highlight cutoff) cannot escape to the frame border through non-bright
    pixels even diagonally, spans at least HEART_MIN_PIXELS pixels, and the
    bright edge sealing it is a single 4-connected group that never touches
    the frame border. Pixels with a 0 or 255 RGB channel serve neither as
    dark heart nor as opening-blocking bright edge.
    """
    left, right, top, bottom = (config[key] for key in ("left", "right", "top", "bottom"))
    cutoff = config["cutoff"]
    height, width = image.shape[:2]
    luma = image @ LUMA
    clean = (image.min(axis=2) >= 1) & (image.max(axis=2) <= 254)
    frame = np.zeros((height, width), dtype=bool)
    frame[top:bottom + 1, left:right + 1] = True
    bright = clean & (luma >= cutoff) & frame
    dark = clean & (luma < cutoff) & frame
    passable = frame & ~bright
    escape = np.zeros((height, width), dtype=bool)
    escape[top, left:right + 1] = True
    escape[bottom, left:right + 1] = True
    escape[top:bottom + 1, left] = True
    escape[top:bottom + 1, right] = True
    escape &= passable
    escape = ndi.binary_propagation(escape, mask=passable, structure=EIGHT_WAY)
    hearts, _ = ndi.label(dark & ~escape, structure=EIGHT_WAY)
    edges, _ = ndi.label(bright, structure=FOUR_WAY)
    border = frame & ~ndi.binary_erosion(frame)
    border_edges = set(np.unique(edges[border]).tolist()) - {0}
    found = []
    for label, box in enumerate(ndi.find_objects(hearts), start=1):
        if box is None:
            continue
        heart = hearts[box] == label
        if heart.sum() < HEART_MIN_PIXELS:
            continue
        rows, cols = box
        y0, y1 = max(rows.start - 1, 0), min(rows.stop + 1, height)
        x0, x1 = max(cols.start - 1, 0), min(cols.stop + 1, width)
        local = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        local[rows.start - y0:rows.stop - y0, cols.start - x0:cols.stop - x0] = heart
        near = ndi.binary_dilation(local, structure=EIGHT_WAY) & ~local
        groups = set(np.unique(edges[y0:y1, x0:x1][near]).tolist()) - {0}
        if len(groups) != 1 or groups & border_edges:
            continue
        edge = edges == groups.pop()
        heart_full = np.zeros((height, width), dtype=bool)
        heart_full[box] = heart
        ys, xs = np.nonzero(heart_full)
        found.append({
            "heart": heart_full, "edge": edge,
            "heart_pixels": int(heart_full.sum()), "edge_pixels": int(edge.sum()),
            "center": (float(xs.mean()), float(ys.mean())),
            "heart_bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            "heart_luma_mean": float(luma[heart_full].mean()),
            "heart_luma_min": float(luma[heart_full].min()),
            "edge_luma_mean": float(luma[edge].mean()),
            "edge_luma_min": float(luma[edge].min()),
        })
    return found


def adopted(image, config):
    """The qualifying droplet the current source cut line crosses, if any."""
    focus = config["focus"]
    crossed = [droplet for droplet in droplets(image, config)
               if droplet["heart"][focus].any() or droplet["edge"][focus].any()]
    if not crossed:
        return None
    return max(crossed, key=lambda droplet: (droplet["heart_pixels"], droplet["edge_pixels"]))


def outline(mask):
    """Boundary pixels of a mask: mask pixels with a 4-neighbor outside it."""
    ys, xs = np.nonzero(mask & ~ndi.binary_erosion(mask, structure=FOUR_WAY))
    return list(zip(xs.tolist(), ys.tolist()))


def droplet_summary(droplet):
    if droplet is None:
        return None
    cx, cy = droplet["center"]
    return {
        "x": round(cx, 2), "y": round(cy, 2),
        "heart_pixels": droplet["heart_pixels"], "edge_pixels": droplet["edge_pixels"],
        "heart_bbox": list(droplet["heart_bbox"]),
        "heart_luma_mean": round(droplet["heart_luma_mean"], 3),
        "heart_luma_min": round(droplet["heart_luma_min"], 3),
        "edge_luma_mean": round(droplet["edge_luma_mean"], 3),
        "edge_luma_min": round(droplet["edge_luma_min"], 3),
    }


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
    droplet = adopted(image, config)
    for p in rows:
        if droplet and droplet["heart"][p["y"], p["x"]]:
            p["droplet"] = "dark_heart"
        elif droplet and droplet["edge"][p["y"], p["x"]]:
            p["droplet"] = "bright_edge"
        else:
            p["droplet"] = "other"
    return {"reading": reading, "profile": rows, "rows": metrics, "focus": config["focus"],
            "droplet": droplet_summary(droplet),
            "area": {"pixels": area, "highlight_pixels": marked, "fraction": marked / area}}


def practice():
    width, height = 780, 466
    cfg = settings({"left": 138, "right": 640, "top": 119, "bottom": 352, "focus": 307,
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
