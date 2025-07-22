import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import product
from data import CCPDDataset
from network import CNN_CTC_model
from torch.optim import Adam
from globals import *
from utils import *
from torchvision import transforms

#Hyperparameters combination
batch_sizes = [64, 32]  
learning_rates = [0.001]
weight_decays = [1e-4, 5e-4]
epochs = [40]

CHAR_LIST = sorted(set(PROVINCES+ALPHABETS+ADS))
PLATE_LENGTH = 8

NUM_CHAR = len(CHAR_LIST) + 1 #since we include the blank character

combinations = product(batch_sizes, learning_rates, weight_decays, epochs)

#executing the training and testing for all the possible combinations to get the best one
for bs, lr, wd, ne in combinations:

    #Hyperparameters
    BATCH_SIZE = bs
    LR = lr
    WEIGHT_DECAY = wd
    NUM_EPOCHS = ne

    SAVE_NAME = f"n_epochs_{NUM_EPOCHS}_bs_{BATCH_SIZE}_LR_{LR}_wd_{WEIGHT_DECAY}"

    print(f"training with {SAVE_NAME}")
    
    model = CNN_CTC_model(num_char=NUM_CHAR, hidden_size=256)
    ctc_loss = nn.CTCLoss(blank=0) 

    preprocess = transforms.Compose([
        transforms.Grayscale(),              # converte in 1 canale
        transforms.Resize((48, 144)),       # adatta a H=48, W=144
        transforms.ToTensor(),              # [C, H, W]
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    
    train_dataloader, val_dataloader, test_dataloader = CCPDDataset.get_dataloaders(base_dir="./dataset", batch_size=BATCH_SIZE, transform=preprocess, collate_fn=custom_collate_2)
     #this optimizer uses stochastic gradient descent and has in input the parameters (weights) from 
    #the pretrained model
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    #optimizer = SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)

    #initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    accuracy_val=[]
    accuracy_train=[]
    total_train_loss=[]
    char_accuracy_train =[]
    char_accuracy_val =[]

    epsilon = 1e-6
    #TRAIN LOOP we are doing fine tuning on the task of recognizing plate
    for e in range(NUM_EPOCHS):
        evaluator = Evaluator()
        model.train()
        train_loss= 0.0
        train_acc = []
        val_acc = []
        train_char_acc =[]
        val_char_acc = []
        B_size = 0
        i=0
        #does the for loop for all the items in the same batch
        for batch in train_dataloader:
            #print(f"Batch {i + 1}/{len(train_dataloader)}")
            images = batch["cropped_image"]
            labels = batch["pdlpr_plate_idx"]
            
            images = [img.to(device) for img in images]
            labels = [lab.to(device) for lab in labels]

            # Stack per batch processing
            images = torch.stack(images)        
            labels = torch.stack(labels)

            #Ctc loss expects a simple list not a 2 dimensional tensor, so all the batch
            #index are flattened into one single list
            flat_labels_list = labels.view(-1)
            #we get the output of the models and apply softmax to turn it into probability 
            output_logits = model(images)                      
            output_probabilities = F.log_softmax(output_logits, dim=2)
            #the output of the model are T vectors for the batch size
            T = output_logits.size(0)
            #get the current batch size
            B_size = images.size(0)
            #creates a tensor the length of the batch size filled with the dimention of the input
            #and the dimension of the output, since ctc requires the lengths because it uses one big
            #vector
            input_lengths = torch.full((B_size,), T, dtype=torch.long).to(device)
            target_lengths = torch.full((B_size,), PLATE_LENGTH, dtype=torch.long).to(device)

            #CTC loss
            loss = ctc_loss(output_probabilities, flat_labels_list, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            evaluator.reset()
            evaluator.update_baseline(output_logits, labels)
            
            ##we take the index of words with the highest probabilities
            metrics = evaluator.compute()
            #metrics for the whole batch
            mean_batch_train_char_acc = metrics["char_accuracy"]
            mean_batch_train_acc = metrics["seq_accuracy"]
            #print(mean_batch_train_char_acc)
            #print(mean_batch_train_acc)
            train_acc.append(mean_batch_train_acc)
            train_char_acc.append(mean_batch_train_char_acc)
            
            i+=1
            
        #compute the mean of the full and character accuracy for training
        #for the whole epoch
        mean_train_acc = sum(train_acc)/len(train_acc)
        mean_train_char_acc = sum(train_char_acc)/len(train_char_acc)
        train_loss = train_loss/B_size

        #append the result to lists in order to plot them
        accuracy_train.append(mean_train_acc)
        char_accuracy_train.append(mean_train_char_acc)
        total_train_loss.append(train_loss)

        j=0
        #Validation phase
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                #print(f"Batch {j + 1}/{len(val_dataloader)}")
                images = batch["cropped_image"]
                labels = batch["pdlpr_plate_idx"]
                
                images = [img.to(device) for img in images]
                labels = [lab.to(device) for lab in labels]

                images = torch.stack(images)         
                labels = torch.stack(labels)

                flat_labels_list = labels.view(-1)  
                 
                output_logits = model(images)
                
                evaluator.reset()
                evaluator.update_baseline(output_logits, labels)
                metrics = evaluator.compute()

                #metrics for the whole batch
                mean_batch_val_char_acc = metrics["char_accuracy"]
                mean_batch_val_acc = metrics["seq_accuracy"]
                #print(mean_batch_val_char_acc)
                #print(mean_batch_val_acc)
                val_acc.append(mean_batch_val_acc)
                val_char_acc.append(mean_batch_val_char_acc)
    
                j+=1

        #compute the mean of the accuracy validation score
        mean_val_acc = sum(val_acc)/len(val_acc)
        mean_val_char_acc = sum(val_char_acc)/len(val_char_acc)

        accuracy_val.append(mean_val_acc)
        char_accuracy_val.append(mean_val_char_acc)

        print(f"Epoch {e +1}/{NUM_EPOCHS} - train loss: {train_loss} - train acc: {mean_train_acc} - train char acc: {mean_train_char_acc} - val acc: {mean_val_acc} --  val char acc: {mean_val_char_acc}" )

    #Saving the model
    torch.save(model.state_dict(), f"models/CNNCTC-{SAVE_NAME}.pth")

    #Plotting the figure for the train and validation
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), accuracy_train, label="train acc", marker='o')
    plt.plot(range(1, NUM_EPOCHS + 1), accuracy_val, label="validation acc", marker='s')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Train and validation plate accuracy per epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"metrics_images/train_validation_CNNCTC-{SAVE_NAME}.png")

    #Plotting the figure for the train and validation
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), char_accuracy_train, label="char train acc", marker='o')
    plt.plot(range(1, NUM_EPOCHS + 1), char_accuracy_val, label="char validation acc", marker='s')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Train and validation character accuracy per epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"metrics_images/char_train_validation_CNNCTC-{SAVE_NAME}.png")

    #getting the last accuracy value for train and validation
    final_train_acc = accuracy_train[-1]
    final_val_acc = accuracy_val[-1]

    final_char_train_acc = char_accuracy_train[-1]
    final_char_val_acc = char_accuracy_val[-1]

    with open(f"results/CNNCTC-{SAVE_NAME}.txt", "w") as f:
        f.write(f"Final train accuracy: {final_train_acc:.4f}\n")
        f.write(f"Final validation accuracy: {final_val_acc:.4f}\n")
        f.write(f"Final character train accuracy: {final_char_train_acc:.4f}\n")
        f.write(f"Final character validation accuracy: {final_char_val_acc:.4f}\n")

  
