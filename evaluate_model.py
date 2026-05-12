import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ======================================
# 1. LOAD MODEL
# ======================================
MODEL_PATH = "model/mobilenetv2_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Label kelas
class_labels = ['Ulat Api', 'Kering Daun', 'Layu Daun', 'Daun Menguning']

# ======================================
# 2. LOAD DATASET TEST (TANPA VALIDATION)
# ======================================
test_dir = "dataset/test"

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# ======================================
# 3. PREDIKSI DATA TEST
# ======================================
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# ======================================
# 4. CETAK AKURASI
# ======================================
accuracy = np.mean(y_pred == y_true) * 100
print(f"\nAkurasi Testing: {accuracy:.2f}%")

# ======================================
# 5. CONFUSION MATRIX
# ======================================
cm = confusion_matrix(y_true, y_pred)
print("\n=== CONFUSION MATRIX ===")
print(cm)

# ======================================
# 6. CLASSIFICATION REPORT
# ======================================
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=class_labels))

# ======================================
# 7. VISUALISASI CONFUSION MATRIX
# ======================================
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix MobileNetV2")

os.makedirs("results", exist_ok=True)
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.show()

print("\nConfusion matrix disimpan ke: results/confusion_matrix.png")
