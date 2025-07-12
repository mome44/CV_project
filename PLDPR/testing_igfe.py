import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from igfe import IGFE

transform = transforms.Compose([
    transforms.Resize((48, 144)),         # Resize esatto richiesto da PDLPR
    transforms.ToTensor(),                # Da PIL a Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizzazione simmetrica
])

def load_images_from_folder(folder_path, max_images=10):
    images = []
    for idx, filename in enumerate(os.listdir(folder_path)):
        if filename.lower().endswith((".jpg", ".png")):
            image = Image.open(os.path.join(folder_path, filename)).convert("RGB")
            image = transform(image)
            images.append(image)
        if len(images) >= max_images:
            break
    return torch.stack(images)  # Shape: [B, 3, 48, 144]

import matplotlib.pyplot as plt

# Mostra la prima feature map del primo esempio
def show_feature_map(tensor):
    feature_map = tensor[0, 0].cpu().numpy()  # Primo canale
    plt.imshow(feature_map, cmap='viridis')
    plt.colorbar()
    plt.title("Feature Map [0,0,:,:]")
    plt.show()


# Supponendo che le immagini siano in ./cropped_plates/
folder_path = "./dataset/crops/test"

print("loading cropped images for IGFE feature extraction.......")
images = load_images_from_folder(folder_path)  # images.shape = [B, 3, 48, 144]

print("calling the IGFE model...")
model = IGFE()
model.eval()  # modalità inference

with torch.no_grad():
    features = model(images)  # Output: [B, 512, 6, 18]
    print("Output shape:", features.shape)

show_feature_map(features)
