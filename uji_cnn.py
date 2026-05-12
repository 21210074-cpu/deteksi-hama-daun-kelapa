import tensorflow as tf

print("✅ TensorFlow berhasil diimport!")
print("Versi TensorFlow:", tf.__version__)

# Cek perangkat yang tersedia (CPU/GPU)
devices = tf.config.list_physical_devices()
print("Perangkat terdeteksi:", devices)
