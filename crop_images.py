import glob
import os.path
import random

import cv2
import numpy as np
from tqdm import tqdm


def main():
    image_list = glob.glob("./data/traindata/traindata/*.jpg")
    mask_list = glob.glob("./data/traindata/traindata/*_mask.png")

    assert len(image_list) == len(mask_list)
    length = len(image_list)
    for i in tqdm(range(length)):
        img_path, mask_path = image_list[i], mask_list[i]
        name1 = os.path.splitext(os.path.split(img_path)[-1])[0]
        name2 = os.path.splitext(os.path.split(mask_path)[-1])[0]
        if name1 + "_mask" != name2:
            print("name mismatch!", "===>", name1, " - ", name2)
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img.shape[0] != mask.shape[0] or img.shape[1] != mask.shape[1]:
            if img.shape[0] == mask.shape[1] and img.shape[1] == mask.shape[0]:
                mask = mask.T
            else:
                print("shape mismatch!", "===>", name1)
                continue

        h, w = img.shape[0], img.shape[1]

        count = 0
        loop_count = 0
        while count < 20 and loop_count < 1000:
            loop_count += 1
            x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
            x2, y2 = x1 + 512, y1 + 512
            if x2 >= w or y2 >= h:
                continue

            croped_mask = mask[y1: y2, x1: x2]

            if np.sum(croped_mask > 0) < 1000:
                continue

            croped_img = img[y1: y2, x1: x2, ...]

            cv2.imwrite("./data/traincrop/img/" + name1 + "_" + str(count) + ".jpg", croped_img)
            cv2.imwrite("./data/traincrop/mask/" + name1 + "_" + str(count) + "_mask" + ".jpg", croped_mask)

            count += 1

        img = cv2.resize(img, (h // 2, w // 2))
        mask = cv2.resize(mask, (h // 2, w // 2))
        h, w = img.shape[0], img.shape[1]

        while count < 40 and loop_count < 1000:
            loop_count += 1
            x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
            x2, y2 = x1 + 512, y1 + 512
            if x2 >= w or y2 >= h:
                continue

            croped_mask = mask[y1: y2, x1: x2]

            if np.sum(croped_mask > 0) < 1500:
                continue

            croped_img = img[y1: y2, x1: x2, ...]

            cv2.imwrite("./data/traincrop/img/" + name1 + "_" + str(count) + ".jpg", croped_img)
            cv2.imwrite("./data/traincrop/mask/" + name1 + "_" + str(count) + "_mask" + ".jpg", croped_mask)

            count += 1

        a = 1
        pass
    # mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
    # img_file = list(self.images_dir.glob(name + '.*'))
    pass

main()