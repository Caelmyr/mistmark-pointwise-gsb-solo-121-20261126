"""Export the real photographed stripe readings and observation bundle."""
from io import BytesIO, StringIO
from zipfile import ZIP_DEFLATED, ZipFile
import csv
import json

from PIL import Image, ImageDraw

from . import mist
from .photo import png


def trace_table(result):
    target = StringIO()
    writer = csv.writer(target)
    writer.writerow(("x_px", "y_px", "red", "green", "blue", "luma", "highlight", "droplet"))
    for point in result["profile"]:
        writer.writerow((point["x"], point["y"], *point["rgb"], f'{point["brightness"]:.3f}',
                         int(point["highlight"]), point.get("droplet", "other")))
    return target.getvalue().encode()


def proof(image, card, result):
    height, width = image.shape[:2]
    droplet = mist.adopted(image, card["settings"]) if result.get("droplet") else None
    output = Image.new("RGB",(max(430,width),height+(88 if droplet else 64)),"#1d3335")
    output.paste(Image.fromarray(image),(0,0))
    draw = ImageDraw.Draw(output)
    cfg = card["settings"]
    draw.rectangle((cfg["left"],cfg["top"],cfg["right"],cfg["bottom"]),outline="#559ea7",width=2)
    if result["profile"]:
        draw.line((cfg["left"],cfg["focus"],cfg["right"],cfg["focus"]),fill="#dccbb5",width=1)
    reading = result["reading"]
    if reading:
        for x, color in ((reading["dark_x"], "#8cc6eb"),(reading["light_x"], "#e99957")):
            draw.ellipse((x-4,cfg["focus"]-4,x+4,cfg["focus"]+4),outline=color,width=2)
        detail = f'Source row {cfg["focus"]} | contrast {reading["contrast"]:.2%} | highlights {reading["highlights"]}'
    else:
        detail = "Current source row is outside the selected frame"
    if droplet:
        draw.point(mist.outline(droplet["edge"]),fill="#7fd4ff")
        draw.point(mist.outline(droplet["heart"]),fill="#ff6a5a")
        cx, cy = (round(v) for v in droplet["center"])
        draw.line((cx-3,cy,cx+3,cy),fill="#ffd166",width=1)
        draw.line((cx,cy-3,cx,cy+3),fill="#ffd166",width=1)
    area = result["area"]
    draw.text((14,height+13),detail,fill="#edf7f3")
    draw.text((14,height+35),f'Area highlights: {area["highlight_pixels"]} / {area["pixels"]}',fill="#f3d4a9")
    if droplet:
        cx, cy = droplet["center"]
        draw.text((14,height+57),f'Droplet @({cx:.0f},{cy:.0f}): dark heart {droplet["heart_pixels"]} px | '
                                 f'bright edge {droplet["edge_pixels"]} px | sealed, clear of frame',fill="#c9e8ff")
    return png(output)


def package(card, original, image, result):
    memory = BytesIO()
    with ZipFile(memory,"w",ZIP_DEFLATED) as stream:
        stream.writestr("glass.png",original)
        stream.writestr("reading.json",json.dumps({"card":card,"reading":result["reading"],
                                                   "droplet":result["droplet"],"area":result["area"]},
                                                ensure_ascii=False,indent=2))
        stream.writestr("glass-profile.csv",trace_table(result))
        stream.writestr("glass-proof.png",proof(image,card,result))
    return memory.getvalue()
