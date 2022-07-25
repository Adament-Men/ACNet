import glob
import os.path
import random

import cv2
import numpy as np
from tqdm import tqdm


def main():
    image_list = sorted(glob.glob("./data/testdata/ourdata/*.jpg"))

    length = len(image_list)
    for i in tqdm(range(length)):
        img_path = image_list[i]
        name1 = os.path.splitext(os.path.split(img_path)[-1])[0]

        img = cv2.imread(img_path)

        img = cv2.resize(img, (1500, 1500))
        h, w = img.shape[0], img.shape[1]

        count = 0
        loop_count = 0
        while count < 50 and loop_count < 1000:
            loop_count += 1
            x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
            x2, y2 = x1 + 512, y1 + 512
            if x2 >= w or y2 >= h:
                continue

            croped_img = img[y1: y2, x1: x2, ...]

            cv2.imwrite("./data/testcrop/ourdata/" + name1 + "_" + str(count) + ".jpg", croped_img)

            count += 1

        # img = cv2.resize(img, (512, 512))
        # cv2.imwrite("./data/testcrop/ourdata/" + name1 + ".jpg", img)

main()