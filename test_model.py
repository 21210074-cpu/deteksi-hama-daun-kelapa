import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# ===============================
# 1️⃣ LOAD MODEL
# ===============================
model = tf.keras.models.load_model("model/mobilenetv2_model.h5")

print("✅ Model berhasil dimuat")

# ===============================
# 2️⃣ LOAD DATA TEST (2025)
# ===============================
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    "dataset/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False   # WAJIB FALSE untuk confusion matrix
)

# ===============================
# 3️⃣ EVALUASI AKURASI
# ===============================
loss, accuracy = model.evaluate(test_data)
print("\n🎯 Test Accuracy (Data 2025):", accuracy)

# ===============================
# 4️⃣ PREDIKSI & CONFUSION MATRIX
# ===============================
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)

print("\n📊 Confusion Matrix:")
print(cm)

# ===============================
# 5️⃣ CLASSIFICATION REPORT
# ===============================
class_names = list(test_data.class_indices.keys())

print("\n📈 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ===============================
# 6️⃣ TAMPILKAN CONFUSION MATRIX
# ===============================
# ===============================
# 6️⃣ VISUALISASI CONFUSION MATRIX
# ===============================
plt.figure(figsize=(6,5))

sns.heatmap(cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',  # warna biar lebih jelas
            xticklabels=["Ulat Api", "Layu Daun", "Kering Daun", "Daun Menguning"],
            yticklabels=["Ulat Api", "Layu Daun", "Kering Daun", "Daun Menguning"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Hasil Pengujian Model")

# simpan gambar HD (WAJIB reviewer)
plt.savefig("results/confusion_matrix.png", dpi=300, bbox_inches='tight')

plt.show()
