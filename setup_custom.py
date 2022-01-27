import os
import random
import re
from PIL import Image, ImageFile
from tqdm import tqdm
import numpy as np
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True

ap = argparse.ArgumentParser()

ap.add_argument("-t", "--train", type=str, required=True,
                help="root path for training data example: ~/training_data/voc/train")

ap.add_argument("-v", "--val", type=str, required=True,
                help="root path for val data example: ~/training_data/voc/test")

ap.add_argument("-i", "--images", type=str, default="JPEGImages",
                help="sub path to images, defaults to JPEGImages")

ap.add_argument("-m", "--masks", type=str, default="SegmentationClassPNG",
                help="sub path to your masks, defaults to SegmentationClassPNG")

ap.add_argument("-o", "--output", type=str, required=True,
                help="path to where the split dataset should be stored")

ap.add_argument("--image-format", dest="image_format", type=str, default="jpg",
                help="image format, defaults to jpg")

ap.add_argument("--mask-format", dest="mask_format", type=str, default="png",
                help="mask format, defaults to png")

ap.add_argument("--delete-originals", dest="delete_originals", action="store_true",
                help="delete the original images after storing them into corresponding folders")


args = vars(ap.parse_args())


# Variables (change if needed)

INPUT_TRAIN_IMAGE_PATH = os.path.join(args['train'], args["images"])
INPUT_TRAIN_MASK_PATH = os.path.join(args['train'], args["masks"])

INPUT_VAL_IMAGE_PATH = os.path.join(args['val'], args["images"])
INPUT_VAL_MASK_PATH = os.path.join(args['val'], args["masks"])

OUTPUT_DATA_PATH = args["output"]
OUTPUT_IMAGE_PATH = OUTPUT_DATA_PATH+'/images'
OUTPUT_MASK_PATH = OUTPUT_DATA_PATH+'/annotations'
IMAGE_FORMAT = '.' + args["image_format"]
MASK_FORMAT = '.' + args["mask_format"]

print(f'Getting training images from {INPUT_TRAIN_IMAGE_PATH} and masks from {INPUT_TRAIN_MASK_PATH}')
print(f'Getting val images from {INPUT_VAL_IMAGE_PATH} and masks from {INPUT_VAL_MASK_PATH}')

# Remove duplicates after adding them to train/val folders
delete_old_images = args["delete_originals"]

if delete_old_images:
    print('Warning, set to delete original images')

# Get all images and masks, sort them and shuffle them to generate data sets.

train_masks = [os.path.splitext(x)[0] for x in os.listdir(
    INPUT_TRAIN_MASK_PATH) if MASK_FORMAT in os.path.splitext(x)[1]]

val_masks = [os.path.splitext(x)[0] for x in os.listdir(
    INPUT_VAL_MASK_PATH) if MASK_FORMAT in os.path.splitext(x)[1]]


train_images = [os.path.splitext(x)[0] for x in os.listdir(
    INPUT_TRAIN_IMAGE_PATH) if IMAGE_FORMAT in os.path.splitext(x)[1] and os.path.splitext(x)[0] in train_masks]

val_images = [os.path.splitext(x)[0] for x in os.listdir(
    INPUT_VAL_IMAGE_PATH) if IMAGE_FORMAT in os.path.splitext(x)[1] and os.path.splitext(x)[0] in val_masks]


# all_images.sort(key=lambda var: [int(x) if x.isdigit() else x
#                                  for x in re.findall(r'[^0-9]|[0-9]+', var)])
# all_masks.sort(key=lambda var: [int(x) if x.isdigit() else x
#                                 for x in re.findall(r'[^0-9]|[0-9]+', var)])


# random.seed(230)
# random.shuffle(all_images)


# # Split images to train and val sets (80% : 20% ratio)

# train_split = int(0.8*len(all_images))

# train_images = all_images[:train_split]
# val_images = all_images[train_split:]


# print(
#     f'SPLIT: {len(train_images)} train and {len(val_images)} validation images!')
# print('-------------------------------------------------------------------------------')


# Generate corresponding mask lists for masks


# train_masks = [f for f in all_masks if f in train_images]
# val_masks = [f for f in all_masks if f in val_images]


# Generate required folders

train_folder = 'training'
val_folder = 'validation'

folders = [train_folder, val_folder]

for folder in folders:
    os.makedirs(os.path.join(OUTPUT_IMAGE_PATH, folder), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_MASK_PATH, folder), exist_ok=True)


# Add train and val images and their masks to corresponding folders


def add_images(dir_name, image):

    full_image_path = f"{INPUT_TRAIN_IMAGE_PATH if dir_name == 'training' else INPUT_VAL_IMAGE_PATH}/{image}{IMAGE_FORMAT}"
    img = Image.open(full_image_path)
    img = img.convert("RGB")
    img.save(OUTPUT_IMAGE_PATH+'/{}'.format(dir_name)+'/'+image+IMAGE_FORMAT)

    if delete_old_images:
        os.remove(full_image_path)


def add_masks(dir_name, image):
    
    full_mask_path = f"{INPUT_TRAIN_MASK_PATH if dir_name == 'training' else INPUT_VAL_MASK_PATH}/{image}{MASK_FORMAT}"
    img = Image.open(full_mask_path)

    img.save(OUTPUT_MASK_PATH+'/{}'.format(dir_name)+'/'+image+MASK_FORMAT)

    if delete_old_images:
        os.remove(full_mask_path)


image_folders = [(train_images, train_folder), (val_images, val_folder)]

mask_folders = [(train_masks, train_folder), (val_masks, val_folder)]

print(
    f'Writing images to the {image_folders[0][1]} and {image_folders[1][1]} folders...')

for folder in image_folders:

    array = folder[0]
    name = [folder[1]] * len(array)

    list(map(add_images, tqdm(name), array))

print(
    f'Writing masks to the {mask_folders[0][1]} and {mask_folders[1][1]} folders...')

for folder in mask_folders:

    array = folder[0]
    name = [folder[1]] * len(array)

    list(map(add_masks, tqdm(name), array))

print('Done!')
