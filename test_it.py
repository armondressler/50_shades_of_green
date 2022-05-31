from dataclasses import dataclass
import os

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
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
    predictions: None

    def prediction_success(self):
        return self.predicted_class == self.actual_class


test_data = []
for leaves_class in class_names:
    for leaf_file in os.scandir(f"{data_dir}/{leaves_class}"):
        img = keras.preprocessing.image.load_img(f"{leaf_file.path}", target_size=(img_height,img_width))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        test_data.append(TestInstance(leaf_file.path, img_array, leaves_class, "", None))

leaves_model = tf.keras.models.load_model("data/custom_model")

for test_instance in test_data:
    predictions = leaves_model.predict(test_instance.image_array)
    test_instance.predictions = predictions
    test_instance.predicted_class = class_names[np.argmax(predictions)]
    print(f"Predicted class: {test_instance.predicted_class}, actual class: {test_instance.actual_class} ({'SUCCESS' if test_instance.predicted_class == test_instance.actual_class else 'FAILURE'})" )

success_instances = [test_instance for test_instance in test_data if test_instance.prediction_success()]
failed_instances = [test_instance for test_instance in test_data if not test_instance.prediction_success()]
confusion_matrix = tf.math.confusion_matrix([class_names.index(test_instance.actual_class) for test_instance in test_data],
    [class_names.index(test_instance.predicted_class) for test_instance in test_data],
    num_classes=len(class_names))
print(f"Accuracy: {len(success_instances)/len(test_data)*100}%")


plt.figure(figsize=(16, 16))
for index, test_instance in enumerate(failed_instances):
    ax = plt.subplot(len(failed_instances), 1, index + 1)
    plt.imshow(test_instance.image_array.numpy().astype("uint8").squeeze())
    plt.title(test_instance.predicted_class)
    plt.axis("off")
plt.savefig(f"{data_dir}/failed_predictions.png")
print(f"Failed predictions stored in: {data_dir}/failed_predictions.png")

plt.figure(figsize=(64,64))
figure, ax = plot_confusion_matrix(conf_mat = np.array(confusion_matrix),
                                   class_names = class_names,
                                   show_absolute = False,
                                   show_normed = True,
                                   colorbar = True)

figure.tight_layout()

plt.savefig(f"{data_dir}/confustion_matrix.png")
print(f"Confusion matrix stored in: {data_dir}/confustion_matrix.png")
