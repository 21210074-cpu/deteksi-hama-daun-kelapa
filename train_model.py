from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import os

# =====================
# Path dataset
# =====================
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# =====================
# Parameter training
# =====================
img_size = 224
batch_size = 16
epochs = 25   # 🔥 dinaikkan biar grafik naik

# =====================
# Augmentasi data
# =====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# =====================
# Load data
# =====================
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# =====================
# Model MobileNetV2
# =====================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size, img_size, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)   # 🔥 biar lebih stabil
predictions = Dense(len(train_data.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# =====================
# Freeze sebagian layer (fine tuning ringan)
# =====================
for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True

# =====================
# Compile
# =====================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =====================
# Callback
# =====================
os.makedirs('model', exist_ok=True)
os.makedirs('results', exist_ok=True)

checkpoint = ModelCheckpoint(
    'model/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# =====================
# Training
# =====================
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs,
    callbacks=[checkpoint, early_stop]
)

# =====================
# Simpan model final
# =====================
model.save('model/final_model.h5')
print("✅ Model berhasil disimpan.")

# =====================
# Simpan history ke JSON (PASTI UPDATE)
# =====================
history_path = "results/training_history.json"

# Hapus history lama kalau ada
if os.path.exists(history_path):
    os.remove(history_path)

with open(history_path, "w") as f:
    json.dump(history.history, f)

print(f"📁 Training history disimpan ke: {history_path}")