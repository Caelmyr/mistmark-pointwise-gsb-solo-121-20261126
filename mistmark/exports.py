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
    output = output.convert("RGBA")
    draw = ImageDraw.Draw(output)
    cfg = card["settings"]
    focus = cfg["focus"]
    draw.rectangle((cfg["left"],cfg["top"],cfg["right"],cfg["bottom"]),outline="#559ea7",width=2)
    if result["profile"]:
        draw.line((cfg["left"],focus,cfg["right"],focus),fill="#dccbb5",width=1)

    segment = result.get("segment")
    dark_color = (140, 198, 235)   # #8cc6eb
    light_color = (233, 153, 87)   # #e99957
    if segment:
        halo = Image.new("RGBA", output.size, (0, 0, 0, 0))
        halo_draw = ImageDraw.Draw(halo)
        for stripe in segment["stripes"]:
            color = light_color if stripe["light"] else dark_color
            halo_draw.rectangle(
                (stripe["left"], focus - 9, stripe["right"], focus + 9),
                fill=color + (64,),
            )
        output = Image.alpha_composite(output, halo)
        draw = ImageDraw.Draw(output)
        for stripe in segment["stripes"]:
            color = light_color if stripe["light"] else dark_color
            draw.rectangle((stripe["left"], focus - 9, stripe["right"], focus + 9),
                           outline=color + (255,), width=2)
        draw.rectangle((segment["left"] - 2, focus - 13,
                        segment["right"] + 2, focus + 13),
                       outline=(119, 224, 205, 255), width=1)

    # RGB 0/255 clipping positions sever runs; mark them on the source line.
    for x in result.get("breaks", []):
        draw.line((x, focus - 12, x, focus + 12), fill=(255, 96, 96, 255), width=1)

    reading = result["reading"]
    if reading:
        for x, color in ((reading["dark_x"], dark_color),(reading["light_x"], light_color)):
            draw.ellipse((x-5,focus-5,x+5,focus+5),outline=color+(255,),width=2)
        if reading.get("adopted"):
            detail = (f'Source row {focus} | adopted segment '
                      f'{reading["segment_left"]}-{reading["segment_right"]} | '
                      f'weakest pair {reading["contrast"]:.2%} | '
                      f'row highlights {reading["highlights"]}')
            segment_line = (f'{reading["stripe_count"]} alternating stripes '
                            f'({segment["light_count"]} light / {segment["dark_count"]} dark) '
                            f'| weakest measured pair drives this reading')
        else:
            detail = (f'Source row {focus} | contrast {reading["contrast"]:.2%} '
                      f'(brightest/darkest pixels) | highlights {reading["highlights"]}')
            segment_line = "No stripe segment meets the width and minimum-contrast rules"
    else:
        detail = "Current source row is outside the selected frame"
        segment_line = ""
    area = result["area"]
    draw.text((14,height+8),detail,fill="#edf7f3")
    draw.text((14,height+26),segment_line,fill="#9fe0d2")
    draw.text((14,height+45),f'Area highlights: {area["highlight_pixels"]} / {area["pixels"]}',fill="#f3d4a9")
    return png(output.convert("RGB"))


def package(card, original, image, result):
    memory = BytesIO()
    with ZipFile(memory,"w",ZIP_DEFLATED) as stream:
        stream.writestr("glass.png",original)
        stream.writestr("reading.json",json.dumps(
            {"card":card,"reading":result["reading"],"area":result["area"],
             "segment":result.get("segment"),"criteria":result.get("criteria"),
             "breaks":result.get("breaks",[])},
            ensure_ascii=False,indent=2))
        stream.writestr("glass-profile.csv",trace_table(result))
        stream.writestr("glass-proof.png",proof(image,card,result))
    return memory.getvalue()
