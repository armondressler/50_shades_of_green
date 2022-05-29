from dataclasses import dataclass
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np

class_names = ['ahorn', 'erdbeer', 'gewaechs', 'loorbeer']
data_dir = "./data/testing"
img_height,img_width=180,180


@dataclass
class TestInstance:
    image_file_path: str
    image_array: None
    actual_class: str
    predicted_class: str

    def prediction_success(self):
        return self.predicted_class == self.actual_class


test_data = []
for leaves_class in class_names:
    for leaf_file in os.scandir(f"{data_dir}/{leaves_class}"):
        img = keras.preprocessing.image.load_img(f"{leaf_file.path}", target_size=(img_height,img_width))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        test_data.append(TestInstance(leaf_file.path, img_array, leaves_class, ""))

leaves_model = tf.keras.models.load_model("data/custom_model")

for test_instance in test_data:
    predictions = leaves_model.predict(test_instance.image_array)
    test_instance.predicted_class = class_names[np.argmax(predictions)]
    print(f"Predicted class: {test_instance.predicted_class}, actual class: {test_instance.actual_class} ({'SUCCESS' if test_instance.predicted_class == test_instance.actual_class else 'FAILURE'})" )



plt.figure(figsize=(16, 16))
success_instances = [test_instance for test_instance in test_data if not test_instance.prediction_success()]
for index, test_instance in enumerate(success_instances):
    ax = plt.subplot(len(success_instances), 1, index + 1)
    plt.imshow(test_instance.image_array.numpy().astype("uint8").squeeze())
    plt.title(test_instance.predicted_class)
    plt.axis("off")
plt.savefig(f"{data_dir}/result.png")