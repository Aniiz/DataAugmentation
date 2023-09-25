import cv2
import os
import math
from matplotlib import pyplot as plt

from utils.read_txt_annotations import read_txt_annotations
from utils.class_colors import class_colors


def visualize_bbox_from_output_folder(output_folder, num_columns=3):
    image_files = [file for file in os.listdir(output_folder) if file.endswith((".jpg", ".png"))]
    num_rows = math.ceil(len(image_files) / num_columns)

    fig = plt.figure(figsize=(10, 10))

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(output_folder, image_file)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        txt_file_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + ".txt")
        bboxes, clss = read_txt_annotations(txt_file_path)

        ax = fig.add_subplot(num_rows, num_columns, i + 1)

        for bbox, class_label in zip(bboxes, clss):
            x_min, y_min, x_max, y_max = bbox
            image_height, image_width, _ = img.shape

            x_min = int(x_min * image_width)
            y_min = int(y_min * image_height)
            x_max = int(x_max * image_width)
            y_max = int(y_max * image_height)

            bbox_color = class_colors.get(class_label, (220, 220, 220)) 

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), bbox_color, 3)

        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

visualize_bbox_from_output_folder("output") #Use "input" or "output" for view the data