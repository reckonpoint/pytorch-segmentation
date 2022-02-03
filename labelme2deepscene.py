#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys
import pprint

from PIL import Image
import numpy as np

import labelme

import color_palette

DEEPSCENE_LABELS = {
    'background': 0, #black
    'trail': 1, #sienna
    'grass': 2, #green
    'vegetation': 3, #olive
    'obstacle': 4, #red
    'sky': 5, #blue
    'water': 6, #aqua
}

PALETTE = color_palette.get_palette_as_array().flatten()

def lblsave(filename, lbl):
    
    if osp.splitext(filename)[1] != ".png":
        filename += ".png"

    lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode="P")
    lbl_pil.putpalette(PALETTE)
    lbl_pil.save(filename)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument(
        "--noviz", help="no visualization", action="store_true"
    )
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "rgb"))
    os.makedirs(osp.join(args.output_dir, "GT_index"))

    print("Creating dataset:", args.output_dir)

    class_names = []
    class_name_to_id = {}

    for name in DEEPSCENE_LABELS:
        class_names.append(name)
        class_name_to_id[name] = DEEPSCENE_LABELS[name]

    next_class_id = max(class_name_to_id.values()) + 1

    with open(args.labels, 'r') as f:
        for line in f.readlines():
            while next_class_id in class_name_to_id.values():
                next_class_id += 1
            label = line.strip()
            if label in class_names:
                continue
            class_names.append(label)
            class_name_to_id[label] = next_class_id

    class_names = tuple(class_names)
    print("class_names: ", class_names)
    
    classes = []
    label_color_palette = color_palette.get_palette()
    for k in class_name_to_id:
        cls_id = class_name_to_id[k]
        classes.append((cls_id, k, label_color_palette[cls_id]))
    print('classes: ', classes)

    with open(osp.join(args.output_dir, "class_names.txt"), "w") as f:
        f.writelines("\n".join(class_names))

    with open(osp.join(args.output_dir, "classes.txt"), "w") as f:
        pprint.pprint(classes, f)

    for filename in glob.glob(osp.join(args.input_dir, "*.json")):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "rgb", base + ".jpg")
        out_png_file = osp.join(
            args.output_dir, "GT_index", base + ".png"
        )

        with open(out_img_file, "wb") as f:
            f.write(label_file.imageData)
        img = labelme.utils.img_data_to_arr(label_file.imageData)

        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        lblsave(out_png_file, lbl)


if __name__ == "__main__":
    main()
