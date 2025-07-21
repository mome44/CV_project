import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import product
from data import CCPDDataset
from network import CNN_CTC_model
from torch.optim import Adam, SGD
from globals import *
from utils import *
from evaluator import Evaluator
from torchvision import transforms

CHAR_LIST = sorted(set(PROVINCES+ALPHABETS+ADS))
PLATE_LENGTH = 8

NUM_CHAR = len(CHAR_LIST) + 1 #since we include the blank character

BATCH_SIZE = 32
LR = 0.001
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 40

SAVE_NAME = f"n_epochs_{NUM_EPOCHS}_bs_{BATCH_SIZE}_LR_{LR}_wd_{WEIGHT_DECAY}_a"
model = CNN_CTC_model(num_char=NUM_CHAR, hidden_size=256)
ctc_loss = nn.CTCLoss(blank=0) 

preprocess = transforms.Compose([
    transforms.Grayscale(),             
    transforms.Resize((48, 144)),       
    transforms.ToTensor(),             
    transforms.Normalize((0.5,), (0.5,))
])

_, _, test_dataloader = CCPDDataset.get_dataloaders(base_dir="./dataset", batch_size=BATCH_SIZE, transform=preprocess)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#TESTING PHASE
if os.path.exists(f"models/CNNCTC-{SAVE_NAME}.pth"):
    model.load_state_dict(torch.load(f"models/CNNCTC-{SAVE_NAME}.pth"))
    model.to(device)
else:
    print("model not found. Please train the model first")
model.eval()
test_acc = []
char_test_acc = []

evaluator = Evaluator()
#here we just loop throught the test data and compute the accuracy scores
with torch.no_grad():
    for batch in test_dataloader:
        images = batch["cropped_image"]
        labels = batch["pdlpr_plate_idx"]
        images = [img.to(device) for img in images]
        labels = [lab.to(device) for lab in labels]
        images = torch.stack(images)         
        labels = torch.stack(labels)
        flat_labels_list = labels.view(-1)  
         
        output_logits = model(images)                      
        output_probabilities = F.log_softmax(output_logits, dim=2)
        evaluator.reset()
        evaluator.update_baseline(output_logits, labels)
        metrics = evaluator.compute()
        #metrics for the whole batch
        mean_batch_test_char_acc = metrics["char_accuracy"]
        mean_batch_test_acc = metrics["seq_accuracy"]
        print(mean_batch_test_acc, mean_batch_test_char_acc)
        test_acc.append(mean_batch_test_acc)
        char_test_acc.append(mean_batch_test_char_acc)

mean_acc = sum(test_acc) / len(test_acc)
mean_char_acc = sum(char_test_acc)/len(char_test_acc)
print(f"Test result accuracy: {mean_acc:.4f}")
print(f"Test result char accuracy: {mean_char_acc:.4f}")
#saving the iou result of the training, validation (last step) and testing
with open(f"results/CNNCTC-test-{SAVE_NAME}.txt", "w") as f:
    f.write(f"Final testing accuracy: {mean_acc:.4f}\n")
    f.write(f"Final testing character accuracy: {mean_char_acc:.4f}\n")
