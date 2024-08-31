import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

def preprocess_data(dataset_path, batch_size=32):
    # ImageDataGenerator for data preprocessing and augmentation
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rescale=1.0/255.0  # Optional: Rescale pixel values if not using a specific preprocessing function
    )

    # Load the dataset
    dataset = datagen.flow_from_directory(
        directory=dataset_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',  # Use 'sparse' if labels are integers, 'categorical' for one-hot encoding
        shuffle=True
    )
    return dataset

def extract_features(cnn_model, dataset):
    features_list = []
    labels_list = []

    # Extract features
    for images, labels in dataset:
        features = cnn_model.predict(images)
        features_list.append(features)
        labels_list.append(labels)

        # Stop after one full epoch if using a generator that loops indefinitely
        if len(features_list) * dataset.batch_size >= dataset.samples:
            break

    features = np.vstack(features_list)
    labels = np.hstack(labels_list)

    df = pd.DataFrame(features)
    df['label'] = labels
    return df

# Example usage (if running this file directly)
if __name__ == "__main__":
    dataset_path = 'path/to/your/dataset'  # Replace with your dataset path
    batch_size = 32

    # Load pre-trained ResNet50 model and remove the top layer
    cnn_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    dataset = preprocess_data(dataset_path, batch_size=batch_size)
    features_df = extract_features(cnn_model, dataset)

    print(features_df.head())
