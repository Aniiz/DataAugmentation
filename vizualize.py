import cv2
import os
import math
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from utils.read_txt_annotations import read_txt_annotations
from utils.class_colors import class_colors


def visualize_bbox_from_output_folder(output_folder, num_columns=3, per_page=9):
    image_files = sorted([file for file in os.listdir(output_folder) if file.endswith((".jpg", ".png"))])
    total = len(image_files)
    num_rows = per_page // num_columns
    total_pages = math.ceil(total / per_page)
    state = {"page": 0}

    def load_image(image_file):
        image_path = os.path.join(output_folder, image_file)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        txt_file_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + ".txt")
        bboxes, clss = read_txt_annotations(txt_file_path)

        image_height, image_width, _ = img.shape
        for bbox, class_label in zip(bboxes, clss):
            x_min, y_min, x_max, y_max = bbox
            x_min = int(x_min * image_width)
            y_min = int(y_min * image_height)
            x_max = int(x_max * image_width)
            y_max = int(y_max * image_height)
            bbox_color = class_colors.get(class_label, (220, 220, 220))
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), bbox_color, 3)

        return img, os.path.splitext(image_file)[0]

    def draw_page(page):
        fig.clf()
        start = page * per_page
        page_files = image_files[start:start + per_page]
        count = len(page_files)
        rows = math.ceil(count / num_columns)

        for i, image_file in enumerate(page_files):
            img, name = load_image(image_file)
            ax = fig.add_subplot(rows, num_columns, i + 1)
            ax.imshow(img)
            ax.set_title(name, fontsize=7)
            ax.axis("off")

        ax_prev = fig.add_axes([0.3, 0.01, 0.15, 0.04])
        ax_next = fig.add_axes([0.55, 0.01, 0.15, 0.04])
        btn_prev = Button(ax_prev, "< Anterior")
        btn_next = Button(ax_next, "Próximo >")

        fig.suptitle(f"Página {page + 1} de {total_pages}  ({total} imagens)", fontsize=10)

        def on_prev(_):
            if state["page"] > 0:
                state["page"] -= 1
                draw_page(state["page"])
                fig.canvas.draw()

        def on_next(_):
            if state["page"] < total_pages - 1:
                state["page"] += 1
                draw_page(state["page"])
                fig.canvas.draw()

        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)
        fig._buttons = (btn_prev, btn_next)

        plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    fig = plt.figure(figsize=(12, 8))
    draw_page(0)
    plt.show()


visualize_bbox_from_output_folder("output") #Use "input" or "output" for view the data
