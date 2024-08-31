import tensorflow as tf
from transformers import TFViTForImageClassification, ViTConfig
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

def create_vit_model(num_classes):
    config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
    config.num_labels = num_classes  # Set the number of classes
    vit_model = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224", config=config)
    return vit_model

def train_vit_model(vit_model, train_dataset, num_epochs=50, learning_rate=0.001):
    optimizer = Adam(learning_rate=learning_rate)
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    metric = SparseCategoricalAccuracy()

    vit_model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric])
    vit_model.fit(train_dataset, epochs=num_epochs)
    print('Finished Training')
    return vit_model
