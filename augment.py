import os
import cv2
import albumentations as A

from utils.read_txt_annotations import read_txt_annotations

def apply_augmentations_and_save(input_folder, output_folder, num_of_imgs):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [file for file in os.listdir(input_folder) if file.endswith((".jpg", ".png"))]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        txt_file_path = os.path.join(input_folder, os.path.splitext(image_file)[0] + ".txt")
        annotations_list, original_class_labels = read_txt_annotations(txt_file_path)

        # To access more transforms, visit: https://albumentations.ai/docs/getting_started/transforms_and_targets/
        transform = A.Compose(
            [
                A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.8),
                A.ToGray(p=0.1),
                A.OneOf([
                    A.Blur(blur_limit=5, p=0.5),
                    A.ColorJitter(p=0.5),
                ], p=0.7),
            ],
            bbox_params=A.BboxParams(format="albumentations", label_fields=["class_labels"])
        )

        images_list = [image]
        saved_bboxes = [annotations_list]

        for i in range(num_of_imgs):
            try:
                augmentations = transform(image=image, bboxes=annotations_list, class_labels=original_class_labels)
                augmented_img = augmentations["image"]
                augmented_bboxes = augmentations["bboxes"]
                augmented_class_labels = augmentations["class_labels"]

                if len(augmented_bboxes) == 0:
                    continue

                images_list.append(augmented_img)
                saved_bboxes.append(augmented_bboxes)

                file_name_prefix, ext = os.path.splitext(image_file)
                output_img_path = os.path.join(output_folder, f"{file_name_prefix}_aug_{i}.jpg")
                cv2.imwrite(output_img_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))

                output_txt_path = os.path.join(output_folder, f"{file_name_prefix}_aug_{i}.txt")
                with open(output_txt_path, "w") as txt_file:
                    for label, bbox in zip(augmented_class_labels, augmented_bboxes):
                        x_min, y_min, x_max, y_max = bbox
                        x_center = (x_min + x_max) / 2.0
                        y_center = (y_min + y_max) / 2.0
                        width = x_max - x_min
                        height = y_max - y_min
                        txt_file.write(f"{label} {x_center} {y_center} {width} {height}\n")

            except ValueError:
                print(f"Erro ao processar o arquivo: {image_file}")
                break

apply_augmentations_and_save(input_folder= "input", output_folder = "output", num_of_imgs = 3 )