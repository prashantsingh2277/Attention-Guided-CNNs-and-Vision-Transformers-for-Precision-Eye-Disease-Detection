import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from agcnn import AttentionCNN
from vit_finetuning import create_vit_model

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  
    return image

def extract_features_agcnn(image, agcnn_model):
    agcnn_model.eval()
    with torch.no_grad():
        features = agcnn_model(image)
    return features

def predict(image_path, agcnn_model_path, vit_model_path, num_classes):
    
    agcnn_model = AttentionCNN()
    agcnn_model.load_state_dict(torch.load(agcnn_model_path))

    vit_model = create_vit_model(num_classes)
    vit_model.load_state_dict(torch.load(vit_model_path))

    image = load_image(image_path)

    features = extract_features_agcnn(image, agcnn_model)

    features_flattened = features.view(features.size(0), -1)
    vit_model.eval()
    with torch.no_grad():
        output = vit_model(features_flattened)
        _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

if __name__ == "__main__":
    image_path = 'path_to_image.jpg'  # Path to the input image
    agcnn_model_path = 'trained_agcnn_model.pth'  # Path to the saved AGCNN model
    vit_model_path = 'trained_vit_model.pth'  # Path to the saved ViT model
    num_classes = 10  # Replace with the actual number of classes in your dataset

    predicted_class = predict(image_path, agcnn_model_path, vit_model_path, num_classes)
    print(f'Predicted Class: {predicted_class}')
