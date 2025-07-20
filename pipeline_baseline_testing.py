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
import matplotlib.pyplot as plt
from data import CCPDDataset
from tqdm import tqdm


CHAR_LIST = sorted(set(PROVINCES+ALPHABETS+ADS))
PLATE_LENGTH = 8

for idx, char in enumerate(CHAR_LIST):
    CHAR_IDX[char] = idx + 1  # start from 1
    IDX_CHAR[idx + 1] = char
IDX_CHAR[0] = '_'  # blank character for CTC

NUM_CHAR = len(CHAR_LIST) + 1 #since we include the blank character

BATCH_SIZE = 32
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
ds = CCPDDataset(base_dir="./dataset", transform=preprocess_dataset)
ds = ds.get_dataset("train")
list_image_paths = ds.get_image_names()
print(f"models/CNNCTC-{SAVE_NAME}_a.pth")
# load pre trained model 
if os.path.exists(f"models/CNNCTC-{SAVE_NAME}.pth"):
    cnn_ctc_model.load_state_dict(torch.load(f"models/CNNCTC-{SAVE_NAME}.pth"))
else:
    print("model not found. Please train the model first")

# crop images with YOLO
base_dir = Path("dataset")
image_paths = Path("dataset/images/test")

evaluator = Evaluator(idx2char=IDX_CHAR)
cnn_ctc_model.eval()
i=0
for image_path in image_paths.glob("*.jpg"):
    if i % 10 == 0:
        print(f"processing image {i+1}/{55000}")
    image_name = image_path.name
    plate_label_path = Path("dataset/labels_pdlpr/test/") / (image_path.stem + ".txt")

    with open(plate_label_path, "r", encoding="utf-8") as f:
        pdlpr_plate_str = f.readline().strip()
    #print(pdlpr_plate_str)

    fields = image_path.stem.split("-")    # image_path.stem is the filename without .jpg
    # Field 2 (index 2) is bbox: format is "x1&y1_x2&y2"
    bbox_part = fields[2]
    corners = bbox_part.split("_")
    x1, y1 = map(int, corners[0].split("&"))
    x2, y2 = map(int, corners[1].split("&"))
    
    true_box = [x1, y1, x2, y2]
    #print(true_box)
    plate_number = fields[4]
    character_id_list = plate_number.split("_")
    plate_id = []
    for c in character_id_list:
        plate_id.append(int(c))
    
    #converting the index from the name to the index from the 
    #unified vocabulary
    plate_id= target_to_index(plate_id)
    #print(plate_id)

    
     
    true_plate_idx = torch.tensor(plate_id, dtype=torch.long).to(device)
    
    image = Image.open(image_path).convert("RGB")

    #candidate_bounding_box = plate_detector(image_path, true_box)
    #print(candidate_bounding_box)

    #iou = compute_iou(candidate_bounding_box, true_box)
    #x1,y1, x2, y2 = candidate_bounding_box
    cropped_image = image.crop((x1, y1, x2, y2))
    #import matplotlib.pyplot as plt
    #plt.imshow(cropped_image, cmap="gray")  # usa cmap="gray" per immagini in scala di grigi
    #plt.title("Cropped Plate")
    #plt.axis("off")
    #plt.show()

    processed_image =preprocess(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_model_output = cnn_ctc_model(processed_image)
        evaluator.reset()
        evaluator.update_baseline(logits_model_output, [true_plate_idx])
        metrics = evaluator.compute()

        char_acc = metrics["char_accuracy"]
        plate_acc = metrics["seq_accuracy"]
        #print(f"  Char Acc: {char_acc:.2f}, Seq Acc: {plate_acc:.2f}\n")
        plate_prediction = evaluator.greedy_decode_idx(logits_model_output)[0]
        #print(plate_prediction)

        plate_string = index_to_target(plate_prediction)
        print(f"predicted_plate: {''.join(plate_string)}, original plate: {pdlpr_plate_str}")
    

    i+=1

    


    

