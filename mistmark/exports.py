"""Export the real photographed stripe readings and observation bundle."""
from io import BytesIO, StringIO
from zipfile import ZIP_DEFLATED, ZipFile
import csv
import json

from PIL import Image, ImageDraw

from .photo import png


def trace_table(result):
    target = StringIO()
    writer = csv.writer(target)
    writer.writerow(("x_px", "y_px", "red", "green", "blue", "luma", "highlight"))
    for point in result["profile"]:
        writer.writerow((point["x"], point["y"], *point["rgb"], f'{point["brightness"]:.3f}', int(point["highlight"])))
    return target.getvalue().encode()


def proof(image, card, result):
    height, width = image.shape[:2]
    output = Image.new("RGB",(max(430,width),height+64),"#1d3335")
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
    area = result["area"]
    draw.text((14,height+13),detail,fill="#edf7f3")
    draw.text((14,height+35),f'Area highlights: {area["highlight_pixels"]} / {area["pixels"]}',fill="#f3d4a9")
    return png(output)


def package(card, original, image, result):
    memory = BytesIO()
    with ZipFile(memory,"w",ZIP_DEFLATED) as stream:
        stream.writestr("glass.png",original)
        stream.writestr("reading.json",json.dumps({"card":card,"reading":result["reading"],"area":result["area"]},
                                                ensure_ascii=False,indent=2))
        stream.writestr("glass-profile.csv",trace_table(result))
        stream.writestr("glass-proof.png",proof(image,card,result))
    return memory.getvalue()
