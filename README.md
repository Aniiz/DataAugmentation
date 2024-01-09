# DataAugmentation
Data Augmentation for yolo object detection.
#### Albumentations [albumentations.ai documentation](https://albumentations.ai/docs/)

----
## How to install
1. Make a venv, access [docs.python](https://docs.python.org/3/tutorial/venv.html)
2. Active venv.
3. Install all packages:
```
pip install -r requirements.txt
```
----
## How to use
1. Put your images and labels in input directory.
2. Run augment.py to augment images:
```
Python augment.py
```
3. Run vizualize.py to see the results:
```
Python vizualize.py
```
----
## How to change the number of generated images
1. Open Augment.py.
2. Change num_of_imgs (line 71): 
```
apply_augmentations_and_save(input_folder= "input", output_folder = "output", num_of_imgs = 3 )
```
----
## How to add more modifiers
1. Access [transformers](https://albumentations.ai/docs/getting_started/transforms_and_targets/)
2. Open Augment.py and add your modifier (line 22):
```
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
```
