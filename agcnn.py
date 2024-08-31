import tensorflow as tf
from tensorflow.keras import layers, models
from spatial_attention_block import SpatialAttentionBlock2D 

def AGCNN(num_classes=2):
    inputs = tf.keras.Input(shape=(224, 224, 3))  # Adjust input shape for 2D (64x64 image with 3 channels)

    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = SpatialAttentionBlock2D(kernel_size=5)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = SpatialAttentionBlock2D(kernel_size=5)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = SpatialAttentionBlock2D(kernel_size=5)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = SpatialAttentionBlock2D(kernel_size=5)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = SpatialAttentionBlock2D(kernel_size=5)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(1024, kernel_size=3, padding='same', activation='relu')(x)
    x = SpatialAttentionBlock2D(kernel_size=5)(x)
    x = layers.MaxPooling2D(pool_size=1)(x)

    x = layers.Conv2D(2048, kernel_size=3, padding='same', activation='relu')(x)
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

    example_input = tf.random.normal((16, 64, 64, 3))  # Example input with 3 channels
    example_labels = tf.random.uniform((16,), minval=0, maxval=2, dtype=tf.int32)

    train_dataset = tf.data.Dataset.from_tensor_slices((example_input, example_labels)).batch(4)
    
    trained_model = train_agcnn_model(model, train_dataset)