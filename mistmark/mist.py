"""Measure actual photographed stripe contrast and highlights on misted glass."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from image_lab.validation import Invalid, integer, number

from .photo import decode, pixels, png, MAX_IMAGE_BYTES


LUMA = np.array([0.2126, 0.7152, 0.0722])

# Stripe-segment criteria.  The canonical persisted keys are
# ``min_width`` / ``max_width`` / ``min_contrast``; the alias tuples let
# callers (and older/newer front-end builds) use equally natural names.
DEFAULT_MIN_WIDTH = 8
DEFAULT_MAX_WIDTH = 120
DEFAULT_MIN_CONTRAST = 0.10

_WIDTH_MIN_KEYS = (
    "min_width", "width_min", "minimum_width", "stripe_min_width", "stripe_width_min",
    "min_stripe_width", "stripe_width_lower", "width_lower", "lower_width",
    "stripe_lower_width", "stripe_min", "min_stripe", "bar_min_width", "bar_width_min",
    "band_min_width", "band_width_min", "wmin",
)
_WIDTH_MAX_KEYS = (
    "max_width", "width_max", "maximum_width", "stripe_max_width", "stripe_width_max",
    "max_stripe_width", "stripe_width_upper", "width_upper", "upper_width",
    "stripe_upper_width", "stripe_max", "max_stripe", "bar_max_width", "bar_width_max",
    "band_max_width", "band_width_max", "wmax",
)
_CONTRAST_KEYS = (
    "min_contrast", "contrast_min", "minimum_contrast", "min_pair_contrast",
    "contrast_threshold", "contrast_floor", "required_contrast",
    "min_contrast_ratio", "weakest_contrast_required",
)
_CONTRAST_PCT_KEYS = (
    "min_contrast_percent", "min_contrast_pct", "contrast_min_percent",
    "minimum_contrast_percent", "contrast_percent",
)


def _pick(value, names, default=None):
    """First non-None entry of ``value`` for one of ``names``."""
    if isinstance(value, dict):
        for name in names:
            item = value.get(name)
            if item is not None:
                return item
    return default


def _contrast_ratio(raw):
    """Accept a 0..1 ratio, a ``"12%"`` string, or a 0..100 percent number."""
    if isinstance(raw, str) and raw.strip().endswith("%"):
        return number(raw.strip()[:-1], "最小对比度", 0, 100) / 100.0
    value = number(raw, "最小对比度", 0, 100)
    if value > 1.0:  # percent entry such as 12 meaning 12%
        value /= 100.0
    return min(1.0, max(0.0, float(value)))


def canonicalize(value):
    """Map recognized setting aliases onto the canonical keys.

    Only keys actually present are returned, so this can be merged over stored
    settings before full validation (otherwise a stale canonical default would
    shadow an incoming alias during a shallow merge).
    """
    if not isinstance(value, dict):
        return {}
    mapped = {}
    for canonical, names in (("min_width", _WIDTH_MIN_KEYS),
                             ("max_width", _WIDTH_MAX_KEYS),
                             ("min_contrast", _CONTRAST_KEYS)):
        for name in names:
            if value.get(name) is not None:
                mapped[canonical] = value[name]
                break
        if canonical not in mapped:
            pct = _pick(value, _CONTRAST_PCT_KEYS, None)
            if canonical == "min_contrast" and pct is not None:
                mapped[canonical] = float(pct) / 100.0
    return mapped


def criteria(config):
    """Resolve stripe criteria from a plain settings/config mapping.

    Missing keys fall back to defaults; present-but-invalid values raise
    :class:`~image_lab.validation.Invalid` just like :func:`settings`.
    """
    config = config if isinstance(config, dict) else {}
    min_width = integer(_pick(config, _WIDTH_MIN_KEYS, DEFAULT_MIN_WIDTH),
                        "每条宽度下限", 1, 100000)
    max_width = integer(_pick(config, _WIDTH_MAX_KEYS, DEFAULT_MAX_WIDTH),
                        "每条宽度上限", 1, 100000)
    if min_width > max_width:
        raise Invalid("每条宽度下限不能大于上限")
    pct = _pick(config, _CONTRAST_PCT_KEYS, None)
    raw_contrast = _pick(config, _CONTRAST_KEYS, None)
    if raw_contrast is None and pct is not None:
        min_contrast = number(pct, "最小对比度", 0, 100) / 100.0
    else:
        min_contrast = _contrast_ratio(DEFAULT_MIN_CONTRAST if raw_contrast is None
                                       else raw_contrast)
    return {"min_width": min_width, "max_width": max_width, "min_contrast": min_contrast}


def settings(value, size):
    width, height = size
    left = integer(value.get("left"), "左界 X", 0, width - 1)
    top = integer(value.get("top"), "上界 Y", 0, height - 1)
    chosen = criteria(value)
    min_width, max_width, min_contrast = (chosen["min_width"], chosen["max_width"],
                                          chosen["min_contrast"])
    config = {
        "left": left, "right": integer(value.get("right"), "右界 X", left, width - 1),
        "top": top, "bottom": integer(value.get("bottom"), "下界 Y", top, height - 1),
        "focus": integer(value.get("focus"), "截线 Y", 0, height - 1),
        "cutoff": integer(value.get("cutoff"), "亮点下限", 1, 254),
        "min_width": min_width, "max_width": max_width, "min_contrast": min_contrast,
    }
    # Read-compatibility aliases; the canonical keys above drive the UI.
    config.update({key: min_width for key in _WIDTH_MIN_KEYS if key != "min_width"})
    config.update({key: max_width for key in _WIDTH_MAX_KEYS if key != "max_width"})
    config.update({key: min_contrast for key in _CONTRAST_KEYS if key != "min_contrast"})
    config.update({key: 100.0 * min_contrast for key in _CONTRAST_PCT_KEYS})
    config["min_contrast_percent"] = 100.0 * min_contrast
    return config


def defaults(size):
    width, height = size
    return settings({
        "left": width // 7, "right": width * 6 // 7, "top": height // 5,
        "bottom": height * 4 // 5, "focus": height // 2, "cutoff": 175,
    }, size)


def contrast(low, high):
    levels = high + low
    return np.divide(high - low, levels, out=np.zeros_like(levels), where=levels > 0)


def pair_contrast(dark, light):
    """(bright median - dark median) / (bright median + dark median)."""
    total = float(light) + float(dark)
    if total <= 0:
        return 0.0
    return float((float(light) - float(dark)) / total)


def adjacent_contrast(a, b):
    """Contrast of two adjacent, necessarily opposite-tone, runs."""
    light = a if a["light"] else b
    dark = b if a["light"] else a
    return pair_contrast(dark["median"], light["median"])


def _stripe_dict(index, light, x0, x1, median):
    center = (x0 + x1) // 2
    width = x1 - x0 + 1
    tone = "light" if light else "dark"
    return {
        "index": index,
        "tone": tone, "kind": tone, "type": tone, "name": tone,
        "light": bool(light), "bright": bool(light), "dark": not light,
        "label": "明" if light else "暗",
        "left": int(x0), "right": int(x1),
        "start": int(x0), "end": int(x1),
        "x0": int(x0), "x1": int(x1),
        "start_x": int(x0), "end_x": int(x1),
        "stop": int(x1 + 1), "end_exclusive": int(x1 + 1),
        "x": int(center), "center": int(center), "center_x": int(center),
        "mid": int(center), "middle": int(center),
        "width": int(width), "length": int(width), "pixels": int(width),
        "brightness": float(median), "median": float(median),
        "luma": float(median), "value": float(median), "level": float(median),
    }


def stripe_runs(line, cutoff, offset_x=0):
    """Run-length encode one cut line into same-tone original-pixel runs.

    A pixel with any RGB channel exactly 0 or 255 is a break: it is not part
    of any stripe and runs on its two sides belong to different blocks.
    """
    rgb = np.asarray(line, dtype=np.uint8)
    if rgb.ndim != 2 or rgb.shape[1] < 3:
        return [], []
    rgb3 = rgb[:, :3]
    luma = rgb3 @ LUMA
    broken = ((rgb3 == 0) | (rgb3 == 255)).any(axis=1)
    bright = luma >= cutoff
    break_x = [int(offset_x + i) for i in range(len(rgb3)) if broken[i]]
    runs = []
    block = 0
    i, n, index = 0, len(rgb3), 0
    while i < n:
        if broken[i]:
            block += 1
            i += 1
            continue
        tone = bool(bright[i])
        j = i
        while (j + 1 < n and not broken[j + 1]
               and bool(bright[j + 1]) == tone):
            j += 1
        stripe = _stripe_dict(index, tone, offset_x + i, offset_x + j,
                              float(np.median(luma[i:j + 1])))
        stripe["block"] = block
        stripe["broken"] = False
        runs.append(stripe)
        index += 1
        i = j + 1
    return runs, break_x


def _pair_dict(left_stripe, right_stripe):
    light = left_stripe if left_stripe["light"] else right_stripe
    dark = right_stripe if left_stripe["light"] else left_stripe
    value = adjacent_contrast(left_stripe, right_stripe)
    return {
        "contrast": value,
        "value": value,
        "ratio": value,
        "light_x": int(light["center"]), "dark_x": int(dark["center"]),
        "high": float(light["median"]), "low": float(dark["median"]),
        "light_median": float(light["median"]),
        "dark_median": float(dark["median"]),
        "light_index": light["index"], "dark_index": dark["index"],
        "indices": [int(left_stripe["index"]), int(right_stripe["index"])],
        "light": light, "dark": dark,
    }


def _segment_dict(chain, y, ordinal):
    pairs = [_pair_dict(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
    weakest = min(pairs, key=lambda pair: pair["contrast"])
    weakest_value = float(weakest["contrast"])
    light_count = sum(1 for stripe in chain if stripe["light"])
    dark_count = len(chain) - light_count
    x0, x1 = chain[0]["left"], chain[-1]["right"]
    count = len(chain)
    segment = {
        "index": ordinal, "ordinal": ordinal, "adopted": False,
        "y": y, "focus": y,
        "left": x0, "right": x1, "start": x0, "end": x1,
        "x0": x0, "x1": x1, "start_x": x0, "end_x": x1,
        "stop": int(x1 + 1), "end_exclusive": int(x1 + 1),
        "offset": x0, "span": int(x1 - x0 + 1), "width": int(x1 - x0 + 1),
        "stripe_count": count, "count": count, "stripes_count": count,
        "bars": count, "number": count,
        "light_count": light_count, "bright_count": light_count,
        "dark_count": dark_count,
        "weakest_contrast": weakest_value, "min_contrast": weakest_value,
        "contrast": weakest_value, "weak_contrast": weakest_value,
        "weakest_contrast_percent": 100.0 * weakest_value,
        "contrast_percent": 100.0 * weakest_value,
        "score": [count, weakest_value],
        "weakest_pair": weakest, "weakest": weakest,
        "worst_pair": weakest, "pairs": pairs, "adjacent": pairs,
        "stripes": chain, "bars_list": chain,
    }
    return segment


def find_segments(line, cutoff=None, min_width=None, max_width=None,
                  min_contrast=None, offset_x=0, y=None, config=None):
    """Return every maximal qualifying alternating stripe segment on a row.

    Qualifying segments have at least two light and two dark continuously
    alternating runs; every run width is within the inclusive width bounds;
    no RGB 0/255 break sits inside; every adjacent pair reaches the minimum
    contrast.  Returned segments are ordered left to right.
    """
    resolved = criteria(config or {})
    if cutoff is None:
        cutoff = (config or {}).get("cutoff", 175)
    if min_width is None:
        min_width = resolved["min_width"]
    if max_width is None:
        max_width = resolved["max_width"]
    if min_contrast is None:
        min_contrast = resolved["min_contrast"]
    cutoff = integer(cutoff, "亮点下限", 1, 254)
    min_width = integer(min_width, "每条宽度下限", 1, 100000)
    max_width = integer(max_width, "每条宽度上限", 1, 100000)
    min_contrast = _contrast_ratio(min_contrast)
    if min_width > max_width:
        raise Invalid("每条宽度下限不能大于上限")

    runs, _ = stripe_runs(line, cutoff, offset_x)
    for run in runs:
        run["width_ok"] = min_width <= run["width"] <= max_width

    segments = []
    i, n, ordinal = 0, len(runs), 0
    while i < n:
        if not runs[i]["width_ok"]:
            i += 1
            continue
        j = i
        while (j + 1 < n and runs[j + 1]["width_ok"]
               and runs[j + 1]["block"] == runs[j]["block"]
               and adjacent_contrast(runs[j], runs[j + 1]) >= min_contrast):
            j += 1
        chain = runs[i:j + 1]
        light_count = sum(1 for run in chain if run["light"])
        dark_count = len(chain) - light_count
        if len(chain) >= 4 and light_count >= 2 and dark_count >= 2:
            segments.append(_segment_dict(chain, y, ordinal))
            ordinal += 1
        i = j + 1
    return segments


def row_segments(image, config):
    """Segments on the configured focus line, or ``[]`` when out of frame."""
    y = config["focus"]
    if not config["top"] <= y <= config["bottom"]:
        return []
    x0, x1 = config["left"], config["right"]
    line = np.asarray(image)[y, x0:x1 + 1]
    return find_segments(line, config=config, cutoff=config.get("cutoff"),
                         offset_x=x0, y=y)


def choose_segment(segments):
    """Balance more captured stripes with a clearer weakest pair.

    Maximise stripe count first, then the weakest-pair contrast; exact ties
    keep the left-most segment (segments arrive in source order).
    """
    if not segments:
        return None
    return max(segments, key=lambda segment: (segment["stripe_count"],
                                              segment["weakest_contrast"]))


def stripe_segment(image, config):
    """Best adopted segment for the configured cut line, or ``None``."""
    best = choose_segment(row_segments(image, config))
    if best is not None:
        best["adopted"] = True
    return best


def segments_from_profile(profile, config=None):
    """Run segmentation from the :func:`profile` point-list representation."""
    config = config if isinstance(config, dict) else {}
    if not profile:
        return []
    line = np.array([point["rgb"] for point in profile], dtype=np.uint8)
    offset_x = int(profile[0].get("x", 0))
    return find_segments(line, config=config, cutoff=config.get("cutoff", 175),
                         offset_x=offset_x, y=config.get("focus"))


def best_from_profile(profile, config=None):
    best = choose_segment(segments_from_profile(profile, config))
    if best is not None:
        best["adopted"] = True
    return best


def select_segment(source=None, config=None, **kwargs):
    """Tolerant entry point accepting an image, a raw RGB line, or a profile."""
    if source is None and "line" in kwargs:
        source = kwargs.pop("line")
    if isinstance(source, list) and source and isinstance(source[0], dict) \
            and "rgb" in source[0]:
        return best_from_profile(source, config)
    array = np.asarray(source)
    if array.ndim == 3:
        if not isinstance(config, dict):
            raise Invalid("缺少凝露设置")
        return stripe_segment(array, config)
    if array.ndim == 2:
        cfg = config if isinstance(config, dict) else {}
        cutoff = kwargs.get("cutoff", cfg.get("cutoff", 175))
        segments = find_segments(
            array, config=cfg, cutoff=cutoff,
            offset_x=kwargs.get("offset_x", cfg.get("left", 0)), y=cfg.get("focus"))
        best = choose_segment(segments)
        if best is not None:
            best["adopted"] = True
        return best
    raise Invalid("无法识别的截线数据")


# Descriptive aliases for callers/tests that name the step differently.
find_segment = select_segment
best_segment = select_segment
adopt_segment = select_segment
pick_segment = select_segment
find_stripe_segment = select_segment
select_stripe_segment = select_segment
best_stripe_segment = select_segment
best_from_line = select_segment
detect_segments = row_segments
stripe_segments = row_segments
measure_stripes = row_segments
segment_line = find_segments
segments_on_row = find_segments


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
    extreme_reading = dict(reading) if reading is not None else None

    segments = row_segments(image, config)
    segment = choose_segment(segments)
    breaks = []
    if rows:
        line = np.asarray(image)[config["focus"], config["left"]:config["right"] + 1]
        _, breaks = stripe_runs(line, config.get("cutoff", 175), config["left"])

    if segment is not None:
        segment["adopted"] = True
        for candidate in segments:
            candidate["adopted"] = candidate is segment
        weak = segment["weakest_pair"]
        if reading is None:
            reading = {}
        # The headline reading is the weakest measured adjacent pair of the
        # adopted segment, replacing the easily reflection-fooled extremes.
        reading.update({
            "adopted": True,
            "source": "stripe_segment",
            "contrast": segment["weakest_contrast"],
            "low": weak["dark_median"],
            "high": weak["light_median"],
            "dark_x": weak["dark_x"],
            "light_x": weak["light_x"],
            "segment_left": segment["left"],
            "segment_right": segment["right"],
            "segment_y": segment["y"],
            "stripe_count": segment["stripe_count"],
            "weakest_contrast": segment["weakest_contrast"],
            "contrast_percent": 100.0 * segment["weakest_contrast"],
        })
    elif reading is not None:
        reading.update({"adopted": False, "source": "row_extremes"})

    area = width * len(metrics)
    marked = sum(row["highlights"] for row in metrics)
    adopted_stripes = segment["stripes"] if segment else []
    result = {
        "reading": reading, "profile": rows, "rows": metrics, "focus": config["focus"],
        "extreme_reading": extreme_reading, "row_reading": extreme_reading,
        "area": {"pixels": area, "highlight_pixels": marked, "fraction": marked / area},
        "segment": segment,
        "segments": segments,
        "best": segment,
        "stripe": segment,
        "stripe_reading": segment,
        "stripe_segment": segment,
        "best_segment": segment,
        "adopted_segment": segment,
        "chosen_segment": segment,
        "selection": segment,
        "candidate_segments": segments,
        "stripes": adopted_stripes,
        "adopted_stripes": adopted_stripes,
        "breaks": breaks,
        "break_points": breaks,
        "adopted": segment is not None,
        "criteria": criteria(config),
    }
    return result


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
