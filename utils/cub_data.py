import numpy as np
import pandas as pd
import os
def load_data(data_root, split):
    # Load image data
    images = pd.read_csv(
        os.path.join(data_root, "CUB_200_2011", "images.txt"),
        sep=" ", names=["image_id", "filepath"],
    )
    image_class_labels = pd.read_csv(
        os.path.join(data_root, "CUB_200_2011", "image_class_labels.txt"),
        sep=" ", names=["image_id", "class_id"],
    )
    train_test_split = pd.read_csv(
        os.path.join(data_root, "CUB_200_2011", "train_test_split.txt"),
        sep=" ", names=["image_id", "is_training_image"],
    )
    classes = pd.read_csv(
        os.path.join(data_root, "CUB_200_2011", "classes.txt"),
        sep=" ", names=["class_id", "class_name"],
    )

    data = images.merge(image_class_labels, on="image_id")
    data = data.merge(train_test_split, on="image_id")
    data = data.merge(classes, on="class_id")
    # Get data split
    if split == "train":
        data = data[data.is_training_image == 1]
    elif split == "valid":
        data = data[data.is_training_image == 0]
    elif split == "all":
        data = data

    data["class_name"] = [class_name.split(".")[1].lower().replace("_", " ") for class_name in data.class_name]

    # Load attribute data
    # image_attribute_labels = pd.read_csv(
    #     os.path.join(data_root, "CUB_200_2011", "attributes", "image_attribute_labels.txt"),
    #     sep=" ", names=["image_id", "attribute_id", "is_present", "certainty_id", "time"],
    # )
    # attributes = pd.read_csv(
    #     os.path.join(data_root, "CUB_200_2011", "attributes", "attributes.txt"),
    #     sep=" ", names=["attribute_id", "attribute_name"]
    # )
    # attributes_info = [attr.split("::") for attr in attributes.attribute_name]
    # attributes_info = np.array([[attr.replace("_", " "), label.replace("_", " ")] for attr, label in attributes_info])
    # attributes["attribute_template"] = attributes_info[:, 0]
    # attributes["attribute_label"] = attributes_info[:, 1]
    # attributes = image_attribute_labels.merge(attributes, on="attribute_id")
    # unique_attributes = attributes.attribute_template.unique()
    return data