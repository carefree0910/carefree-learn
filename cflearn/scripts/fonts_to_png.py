import os

import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from typing import Any
from typing import Tuple
from cftool.dist import Parallel


tight = False
padding = 0.1
"""
Expected file structure:
-- /path/to/your/fonts/folder
 |- font_type0
  |- font_file0
  |- font_file1
  ...
 |- font_type1
  |- font_file0
  |- font_file1
  ...

A typical use case is to convert opentype / truetype fonts:
-- /path/to/your/fonts/folder
 |- opentype
  |- xxx.otf
  |- xxx.otf
  ...
 |- truetype
  |- xxx.ttf
  |- xxx.ttf
  ...

"""
fonts_folder = "/path/to/your/fonts/folder"

num_jobs = 32
resolution = 512
test_resolution = 32
export_folder = f"export{'-tight' if tight else ''}"

all_folder = os.path.join(export_folder, "all")
split_folder = os.path.join(export_folder, "split")
lower = "abcdefghijklmnopqrstuvwxyz"
upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
alphabet = lower + upper
limit = test_resolution * (1.0 - padding)
width = resolution * 13
height = resolution * 4
os.makedirs(all_folder, exist_ok=True)
os.makedirs(split_folder, exist_ok=True)


def filter_fn(name: str) -> bool:
    if name.startswith("ZillaSlabHighlight"):
        return False
    return True


def get_border(mask: np.ndarray, axis: int) -> int:
    indices = np.argmax(mask, axis=axis)
    first = mask[0] if axis == 0 else mask[:, 0]
    return (indices[indices != 0 | first]).min().item()


def get_bbox(img: np.ndarray) -> Tuple[int, int, int, int]:
    mask = img > 50
    left = get_border(mask, axis=1)
    right = mask.shape[1] - get_border(mask[:, ::-1], axis=1)
    top = get_border(mask, axis=0)
    bottom = mask.shape[0] - get_border(mask[::-1], axis=0)
    return left, right, top, bottom


def get_bbox_from(char: str, font: Any, res: int) -> Tuple[int, int, int, int]:
    image = Image.new("RGB", [res * 2, res * 2])
    draw = ImageDraw.Draw(image)
    draw.text((res, 0), char, font=font)
    left, right, top, bottom = get_bbox(np.array(image.convert("L")))
    return left - res, right - res, top, bottom


def get_font(char: str, font_path: str) -> Tuple[Any, float, float]:
    font_size = 1.0
    while True:
        font = ImageFont.truetype(font_path, int(round(font_size)))
        fw, fh = font.getsize(char)
        if fw >= limit or fh >= limit:
            break
        font_size += 1.0
    left, right, top, bottom = get_bbox_from(char, font, test_resolution)
    ratio = resolution / test_resolution
    if tight:
        span = max(right - left, bottom - top)
        span_ratio = limit / span
        font_size *= span_ratio
        font = ImageFont.truetype(font_path, int(round(font_size)))
        res = int(round(test_resolution * span_ratio))
        left, right, top, bottom = get_bbox_from(char, font, res)
    x_offset = 0.5 * (test_resolution - (left + right))
    y_offset = 0.5 * (test_resolution - (top + bottom))
    font = ImageFont.truetype(font_path, int(round(font_size * ratio)))
    x_offset *= ratio
    y_offset *= ratio
    return font, x_offset, y_offset


def main_task(folder: str, file: str) -> None:
    name = os.path.splitext(file)[0]
    if not filter_fn(name):
        return None
    export_file = f"{name}.png"
    export_path = os.path.join(all_folder, export_file)
    if os.path.isfile(export_path):
        return None
    try:
        x_offsets = set()
        y_offsets = set()
        path = os.path.join(folder, file)
        image = Image.new("RGB", [width, height])
        draw = ImageDraw.Draw(image)
        for i in range(52):
            ix, iy = i % 13, i // 13
            x, y = ix * resolution, iy * resolution
            char = (lower if i < 26 else upper)[i % 26]
            font, x_offset, y_offset = get_font(char, path)
            x_offsets.add(x_offset)
            y_offsets.add(y_offset)
            draw.text((x + x_offset, y + y_offset), char, font=font)
        if max(len(x_offsets), len(y_offsets)) <= 5:
            return None
        image.save(export_path)
    except Exception as err:
        print(f"> failed to export {file} : {err}")


def main() -> None:
    folders = []
    files = []
    for font_type in ["opentype", "truetype"]:
        type_folder = os.path.join(fonts_folder, font_type)
        for file in sorted(os.listdir(type_folder)):
            folders.append(type_folder)
            files.append(file)
    shuffle_indices = np.random.permutation(len(folders))
    folders = [folders[i] for i in shuffle_indices]
    files = [files[i] for i in shuffle_indices]
    Parallel(num_jobs).grouped(main_task, folders, files)


def split_task(file: str) -> None:
    name = os.path.splitext(file)[0]
    name_folder = os.path.join(split_folder, name)
    os.makedirs(name_folder, exist_ok=True)
    img = np.array(Image.open(os.path.join(all_folder, file)))
    for i in range(4):
        for j in range(13):
            char = alphabet[i * 13 + j]
            path = os.path.join(name_folder, f"{char}.png")
            x, y = j * resolution, i * resolution
            ij_img = img[y : y + resolution, x : x + resolution]
            Image.fromarray(ij_img).save(path)


def split() -> None:
    Parallel(num_jobs).grouped(split_task, os.listdir(all_folder))


if __name__ == "__main__":
    main()
    split()
