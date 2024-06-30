# -*- coding: utf-8 -*-
"""
Created on Fri May 24 00:21:39 2024

@author: Cihan
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt

# Veri artırma ve ön işleme
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Pixel değerlerini [0, 1] aralığına ölçeklendirme
    rotation_range=20,  # Rastgele rotasyon
    width_shift_range=0.2,  # Yatayda rastgele kaydırma
    height_shift_range=0.2,  # Dikeyde rastgele kaydırma
    shear_range=0.2,  # Şeğirme dönüşümü
    zoom_range=0.2,  # Rastgele yakınlaştırma
    horizontal_flip=True,  # Yatayda rastgele çevirme
    fill_mode='nearest',  # Doldurma stratejisi
    validation_split=0.2  # Eğitim ve doğrulama verisi ayırımı
)

# Veri seti yolu
dataset_dir = os.path.expanduser('C:/Users/Cihan/Desktop/archive/data')

# Eğitim verilerinin yüklenmesi
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),  # Görüntülerin yeniden boyutlandırılacağı hedef boyut
    batch_size=32,  # Batch boyutu
    class_mode='categorical',  # Sınıflandırma tipi
    subset='training'  # Eğitim verisi
)

# Doğrulama verilerinin yüklenmesi
validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Doğrulama verisi
)

# Hiperparametreler
input_shape = (128, 128, 3)  # Girdi görüntü boyutu (yükseklik, genişlik, kanal sayısı)
num_classes = 4  # Sınıf sayısı

# Model oluşturma
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))  # Sınıflandırma katmanı
# Modelin özeti
model.summary()


# Modelin eğitimi
history = model.fit(train_generator, epochs=15, validation_data=validation_generator)

# Eğitim ve doğrulama kaybı
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Eğitim ve doğrulama doğruluğu
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Modelin değerlendirilmesi
test_loss, test_acc = model.evaluate(validation_generator)
print(f'Test accuracy: {test_acc}')

# Modelin kaydedilmesi
model.save('satellite_classification_model.h5')
