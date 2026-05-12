import json
import matplotlib.pyplot as plt
import os

# Pastikan folder results ada
history_path = "results/training_history.json"

if not os.path.exists(history_path):
    print("❌ File training_history.json tidak ditemukan!")
    print("Jalankan dulu: python train_model.py")
    exit()

# Baca history dari file JSON
with open(history_path, "r") as f:
    history = json.load(f)

# Buat grafik
epochs = range(1, len(history["accuracy"]) + 1)

# Plot Accuracy
plt.figure(figsize=(10,5))
plt.plot(epochs, history["accuracy"], label="Training Accuracy")
plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/accuracy_from_json.png", dpi=300)
plt.close()

# Plot Loss
plt.figure(figsize=(10,5))
plt.plot(epochs, history["loss"], label="Training Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/loss_from_json.png", dpi=300)
plt.close()

print("✅ Grafik berhasil dibuat:")
print("- results/accuracy_from_json.png")
print("- results/loss_from_json.png")
