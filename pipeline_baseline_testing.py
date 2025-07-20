from ultralytics import YOLO
from pathlib import Path
import torch
import os
from PIL import Image  
from evaluator import Evaluator
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from globals import *
from utils import *
from network import CNN_CTC_model
from ultralytics import YOLO
import matplotlib.pyplot as plt
from data import CCPDDataset


def crop_image_yolo(yolo_model, image):
    yolo_results = yolo_model(image)[0]
    detection = yolo_results.boxes.xyxy[0]
    print(detection)
    # Se non ci sono targhe rilevate
    if detection.shape[0] == 0:
        return []

    x1, y1, x2, y2 = detection.tolist()
    x1, y1, x2, y2 = map(int, detection)
    
    cropped_img = image.crop((x1, y1, x2, y2))

    plt.imshow(cropped_img)
    plt.title("Targa rilevata (crop YOLO)")
    plt.axis("off")
    plt.show()

    return cropped_img
    
    ##code to convert to plate tensor
    #with torch.no_grad():
    #    logits = pdlpr_model(plate_tensor)
    #    output_probabilities = F.log_softmax(logits, dim=2)
    #    predictions = torch.argmax(output_probabilities, dim=2)
    #    pred_text = index_to_target(logits)
    #
    #return predicted_plate

#def baseline_pipeline_prediction(cnnctc_model, image_path):
#    return predicted_plate

# loading yolo model
yolo_model = YOLO("runs/train/yolov5_epochs20_bs8_lr0.001_imgs640/weights/best.pt")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

evaluator = Evaluator()

CHAR_LIST = sorted(set(PROVINCES+ALPHABETS+ADS))
PLATE_LENGTH = 8

NUM_CHAR = len(CHAR_LIST) + 1 #since we include the blank character

BATCH_SIZE = 36
LR = 0.001
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 60
SAVE_NAME = f"n_epochs_{NUM_EPOCHS}_bs_{BATCH_SIZE}_LR_{LR}_wd_{WEIGHT_DECAY}_a"

cnn_ctc_model = CNN_CTC_model(num_char=NUM_CHAR, hidden_size=256)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
preprocess = transforms.Compose([
    transforms.Grayscale(),             
    transforms.Resize((48, 144)),       
    transforms.ToTensor(),             
    transforms.Normalize((0.5,), (0.5,))
])

preprocess_dataset = transforms.Compose([                   
    transforms.ToTensor()
])
dataloader, _, _= CCPDDataset.get_dataloaders(base_dir="./dataset", batch_size=1, transform=preprocess_dataset)

# load pre trained model 
if os.path.exists(f"models/CNNCTC-{SAVE_NAME}.pth"):
    cnn_ctc_model.load_state_dict(torch.load(f"models/CNNCTC-{SAVE_NAME}.pth"))
else:
    print("model not found. Please train the model first")



# crop images with YOLO
base_dir = Path("dataset")
image_paths = Path("dataset/images/train")

evaluator = Evaluator()
i = 0
for item in dataloader:
    if i % 10 == 0:
        print(f"processing image {i+1}/{len(dataloader)}")
    image_tensor = item["full_image"]
    yolo_target = item['yolo_bbox_label']
    plate_target = item['pdlpr_plate_idx']

    img_pil = to_pil_image(image_tensor)

    image_tensor = image_tensor.transforms.Resize((640, 640))
    cropped_image = crop_image_yolo(yolo_model, image)
    #apply transformation to the image for the 
    processed_image = preprocess(cropped_image).unsqueeze(0).to(device)

    logits_model_output = cnn_ctc_model(processed_image)
    


    

