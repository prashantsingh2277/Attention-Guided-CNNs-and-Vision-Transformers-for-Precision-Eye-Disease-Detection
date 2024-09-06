import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
from spatial_attention_block import SpatialAttentionBlock2D
import SE



def preprocess_fundus_image(image):
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized / 255.0
    img_gray = cv2.cvtColor(image_normalized, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    img_blurred = cv2.GaussianBlur(img_clahe, (5, 5), 0)
    img_processed = np.stack([img_blurred] * 3, axis=-1)

    return img_processed

def preprocess_dataset(image, label):
    image = tf.numpy_function(preprocess_fundus_image, [image], tf.float32)
    return image, label

def AGCNN(num_classes=2):
    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = SE.se_block(x)
    x = SpatialAttentionBlock2D(kernel_size=5)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = SE.se_block(x)
    x = SpatialAttentionBlock2D(kernel_size=5)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = SE.se_block(x)
    x = SpatialAttentionBlock2D(kernel_size=5)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = SE.se_block(x)
    x = SpatialAttentionBlock2D(kernel_size=5)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = SE.se_block(x)
    x = SpatialAttentionBlock2D(kernel_size=5)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(1024, kernel_size=3, padding='same', activation='relu')(x)
    x = SE.se_block(x)
    x = SpatialAttentionBlock2D(kernel_size=5)(x)
    x = layers.MaxPooling2D(pool_size=1)(x)

    x = layers.Conv2D(2048, kernel_size=3, padding='same', activation='relu')(x)
    x = SE.se_block(x)
    x = SpatialAttentionBlock2D(kernel_size=5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.MaxPooling2D(pool_size=1)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def train_agcnn_model(model, train_dataset, num_epochs=10, learning_rate=0.001):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_dataset, epochs=num_epochs)
    return model

if __name__ == "__main__":
    model = AGCNN(num_classes=2)
    model.summary()

    example_input = np.random.rand(16, 224, 224, 3).astype(np.float32)
    example_labels = np.random.randint(0, 2, size=(16,))
    train_dataset = tf.data.Dataset.from_tensor_slices((example_input, example_labels))
    train_dataset = train_dataset.map(preprocess_dataset).batch(4)
    trained_model = train_agcnn_model(model, train_dataset)
    model_save_path = 'agcnn.h5'
    trained_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
