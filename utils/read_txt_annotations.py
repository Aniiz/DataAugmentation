def read_txt_annotations(file_path):
    annotations_list = []
    class_labels_list = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if line:
                class_label = int(line[0])
                x_center, y_center, width, height = map(float, line[1:])
                x_min = (x_center - width / 2.0)
                y_min = (y_center - height / 2.0)
                x_max = (x_center + width / 2.0)
                y_max = (y_center + height / 2.0)
                annotation = [x_min, y_min, x_max, y_max]

                annotations_list.append(annotation)
                class_labels_list.append(class_label)

    return annotations_list, class_labels_list