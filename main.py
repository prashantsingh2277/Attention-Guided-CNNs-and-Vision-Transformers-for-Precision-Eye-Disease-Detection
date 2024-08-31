import tensorflow as tf
import numpy as np
import pandas as pd
from vit_finetuning import create_vit_model, train_vit_model
from agcnn import AGCNN
from data_preparation import preprocess_data, extract_features
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print(f"Using device: {device}") 

train_dataset = preprocess_data(r"D:\EYE Disease\glucoma\Drishti-GS\Training-20211018T055246Z-001\Training\Images")

agcnn_model = AGCNN(num_classes=len(train_dataset.class_indices))
agcnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

agcnn_model.fit(train_dataset, epochs=10)
agcnn_model.save('trained_agcnn_model.h5')
print("AGCNN model saved successfully as 'trained_agcnn_model.h5'")

df = extract_features(agcnn_model, train_dataset)

def df_to_dataset(df, batch_size=32):
    features = df.iloc[:, :-1].values
    labels = df['label'].values
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size).shuffle(buffer_size=len(df))
    return dataset

df_loader = df_to_dataset(df)

num_classes = len(df['label'].unique())
vit_model = create_vit_model(num_classes)
vit_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

vit_model.fit(df_loader, epochs=10)
vit_model.save('trained_vit_model.h5')
print("ViT model saved successfully as 'trained_vit_model.h5'")
