import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Load trained CNN model
model = tf.keras.models.load_model("model.h5")

# Image settings
img_size = 64
batch_size = 32

# Load test dataset
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

print("Generating Predictions...")

# Predict test data
predictions = model.predict(test_data)

# Convert probabilities to class labels
y_pred = (predictions > 0.5).astype(int)

# True labels
y_true = test_data.classes

# Generate Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Alert", "Drowsy"],
    yticklabels=["Alert", "Drowsy"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - CNN Model")

# Save image
plt.savefig("confusion_matrix_cnn.png")

plt.show()

# Print classification report
print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Alert", "Drowsy"]
    )
)

print("\nConfusion matrix saved as confusion_matrix_cnn.png")
