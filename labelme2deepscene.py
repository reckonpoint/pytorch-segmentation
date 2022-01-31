#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import numpy as np

import labelme

CLASS_MAP = {
    'background': 0, #black
    'obstacle': 1, #maroon
    'grass': 2, #green
    'vegetation': 3, #olive
    'sky': 4, #navy
    'water': 6, #teal
    'road': 7, #grey
    'trail': 8, #brown
}

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

    class_map_max_value =  max(CLASS_MAP.values())
    class_names = ['background']
    class_name_to_id = {'background': 0}
    next_class_id = 5

    with open(args.labels, 'r') as f:
        for i, line in enumerate(f.readlines()):
            while next_class_id in CLASS_MAP.values() or next_class_id in class_name_to_id.values():
                next_class_id += 1
            label = line.strip()
            if label in class_names:
                continue
            class_names.append(label)
            class_name_to_id[label] = CLASS_MAP[label] if label in CLASS_MAP else next_class_id

    class_names = tuple(class_names)
    print("class_names: ", class_names)
    print('classes: ', class_name_to_id)

    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

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
        labelme.utils.lblsave(out_png_file, lbl)


if __name__ == "__main__":
    main()
