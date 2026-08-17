import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import json


dataset_path = "" # This is the path to your dataset

train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split = 0.2,
    subset = "training",
    seed = 42,
    image_size = (224, 224),
    batch_size = 32,
    shuffle = True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split = 0.2,
    subset = "validation",
    seed = 42,
    image_size = (224, 224),
    batch_size = 32,
    shuffle = False
)

class_names = (train_dataset.class_names)

with open("class_names.json", "w") as file:
    json.dump(class_names, file)

'''
for images, labels in train_dataset.take(1):
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
'''


data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ]
)

conv_base = tf.keras.applications.EfficientNetB1(
    weights = "imagenet",
    include_top = False,
    input_shape = (224, 224, 3)
)
conv_base.trainable = False


inputs = tf.keras.Input(shape = (224, 224, 3))
x = data_augmentation(inputs)
x = conv_base(x, training = False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(len(train_dataset.class_names), activation = "softmax")(x)
model = tf.keras.Model(inputs, outputs)


model.compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "rmsprop", #tf.keras.optimizers.Adam(learning_rate = 1e-4),
    metrics = ["accuracy"]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath = "", # This is the path where you want to save the best model
        save_best_only = True,
        monitor = "val_loss"
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor = "val_loss",
        factor = 0.2,
        patience = 3,
        min_lr = 1e-7
    )
]

history = model.fit(
    train_dataset,
    validation_data = validation_dataset,
    epochs = 20,
    callbacks = callbacks
)


plt.plot(history.history["accuracy"], "o", label = "Training Accuracy")
plt.plot(history.history["val_accuracy"], label = "Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history["loss"], "o", label = "Training Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()