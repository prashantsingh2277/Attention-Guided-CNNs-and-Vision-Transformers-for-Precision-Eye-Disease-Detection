import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

def preprocess_data(dataset_path, batch_size=32):
    def custom_preprocess_image(image):
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized / 255.0
        img_gray = cv2.cvtColor(image_normalized, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)
        img_blurred = cv2.GaussianBlur(img_clahe, (5, 5), 0)
        img_processed = np.stack([img_blurred] * 3, axis=-1)
        return img_processed

    datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocess_image,
        rescale=1.0/255.0
    )

    dataset = datagen.flow_from_directory(
        directory=dataset_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=True
    )
    return dataset

def extract_features(cnn_model, dataset):
    features_list = []
    labels_list = []

    for images, labels in dataset:
        features = cnn_model.predict(images)
        features_list.append(features)
        labels_list.append(labels)
        
        if len(features_list) * dataset.batch_size >= dataset.samples:
            break

    features = np.vstack(features_list)
    labels = np.hstack(labels_list)

    df = pd.DataFrame(features)
    df['label'] = labels
    return df
    
if __name__ == "__main__":
    dataset_path = 'path/to/your/dataset'
    batch_size = 32

    model_path = 'agcnn.h5'
    cnn_model = tf.keras.models.load_model(model_path, custom_objects={'SpatialAttentionBlock2D': SpatialAttentionBlock2D, 'se_block': SE.se_block})

    dataset = preprocess_data(dataset_path, batch_size=batch_size)
    features_df = extract_features(cnn_model, dataset)

    print(features_df.head())
