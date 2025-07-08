import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# === 1. Load the saved model ===
model = load_model('flower_classifier_model.h5')

# === 2. Load and preprocess a new flower image ===
img_path = 'image.jpg'  # Change this to your actual image path
img = image.load_img(img_path, target_size=(150, 150))  # Same size as training
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize

# === 3. Make prediction ===
pred = model.predict(img_array)

# === 4. Get class label ===
# You need to provide class labels in the same order as during training
class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  # example list
predicted_class = class_labels[np.argmax(pred)]
confidence = np.max(pred) * 100

print(f"Predicted Flower: {predicted_class} ({confidence:.2f}% confidence)")

# === 5. Show image with prediction ===
plt.imshow(img)
plt.title(f"{predicted_class} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
