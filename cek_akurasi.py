from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path dataset & model
test_dir = 'dataset/test'
model_path = 'model/mobilenetv2_model.h5'

# Load model
model = load_model(model_path)

# Siapkan data uji
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# Evaluasi model
loss, acc = model.evaluate(test_data)
print(f"Akurasi model: {acc * 100:.2f}%")
