import tensorflow as tf
import numpy as np
import json


model = tf.keras.models.load_model(
    "" # This is the path to your trained model
)

json_path = "" # This is the path to your class_names.json file

with open(json_path, "r") as file:
    class_names = json.load(file)


def predict_image(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size = (320, 320)
    )

    image_array = tf.keras.utils.img_to_array(image)

    image_array = np.expand_dims(
        image_array,
        axis = 0
    )

    predictions = model.predict(
        image_array,
        verbose = 0
    )

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    return predicted_class, confidence


image_path = "" # This is the path to the image you want to predict

prediction, confidence = predict_image(image_path)

print("Prediction:", prediction)
print(f"Confidence: {confidence * 100:.2f}%")