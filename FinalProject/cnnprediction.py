import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, Model


# Load the trained model
model = keras.models.load_model('model.keras')

# Initialize the data generator for prediction images (only rescaling)
predict_datagen = ImageDataGenerator(rescale=1./255)

# Set the path to your prediction images
prediction_dir = 'predictionImages/prediction/'

# Check if the directory exists and has images
if not os.path.exists(prediction_dir) or not os.listdir(prediction_dir):
    raise ValueError(f"No images found in directory {prediction_dir}. Please check the path and contents.")

predict_generator = predict_datagen.flow_from_directory(
    directory=prediction_dir,
    target_size=(150, 150),
    batch_size=1,
    class_mode=None,
    shuffle=False
)

if predict_generator.samples == 0:
    raise ValueError("No images found by the ImageDataGenerator. Check your directory structure.")

predictions = model.predict(predict_generator, steps=len(predict_generator))

predicted_class_indices = np.argmax(predictions, axis=1)

class_labels = ['brownie','cake','cherry','eggs','miso soup','soda','steak', 'sundae', 'tree', 'watermelon']

predicted_class_labels = [class_labels[idx] for idx in predicted_class_indices]

num_images = 16


fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(20, 3))

num_images = min(num_images, len(predict_generator.filenames))

for i in range(num_images):
    ax = axes[i]
    path = os.path.join(prediction_dir, predict_generator.filenames[i])
    img = plt.imread(path)
    ax.imshow(img)
    ax.set_title(f"Pred: {predicted_class_labels[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()