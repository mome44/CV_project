import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import product
from data import RecognitionDataset
from network import CNN_CTC_model
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from globals import *
from utils import *


#Hyperparameters combination
batch_sizes = [64, 128]
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

    SAVE_NAME = f"n_epochs_{NUM_EPOCHS}_bs_{BATCH_SIZE}_LR_{LR}_wd_{WEIGHT_DECAY}_a"

    print(f"training with {SAVE_NAME}")
    
    model = CNN_CTC_model(num_char=NUM_CHAR, hidden_size=256)
    ctc_loss = nn.CTCLoss(blank=0) 
    train_dataset = RecognitionDataset("dataset/crops/train", "dataset/labels_pdlpr/train")
    val_dataset = RecognitionDataset("dataset/crops/val", "dataset/labels_pdlpr/val")
    test_dataset = RecognitionDataset("dataset/crops/test", "dataset/labels_pdlpr/test")


    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
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
        model.train()
        train_loss= 0.0
        train_acc = []
        val_acc = []
        train_char_acc =[]
        val_char_acc = []
        B_size = 0
        i=0
        #does the for loop for all the items in the same batch
        for images, labels in train_dataloader:
            #print(f"Batch {i + 1}/{len(train_dataloader)}")
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

            #we take the index of words with the highest probabilities
            predictions = torch.argmax(output_probabilities, dim=2)
            predictions= predictions.transpose(0, 1)
            final_predictions = []
            #iterate for each prediction array in the batch
            for prediction in predictions:
                reduced = []
                before = 0
                for t_index in prediction:
                    t_index = t_index.item()
                    if t_index != 0 and t_index != before:
                        #append the index only if it is not zero and it is different than before
                        reduced.append(t_index)
                    before = t_index
                final_predictions.append(reduced)
            
            
            
            #computing the metrics for the plates
            for pred_idx_list, label in zip(final_predictions, labels):
                label_list = label.tolist()
                #checking if the two list are equal
                if pred_idx_list == label_list:
                    train_acc.append(1)
                else:
                    train_acc.append(0)
                char_correct = 0
                #number of same characters
                for pred_idx, label_idx in zip(pred_idx_list, label):
                    if pred_idx == label_idx:
                        char_correct += 1
                mean_char = char_correct/PLATE_LENGTH
                train_char_acc.append(mean_char)
            i+=1
            
        #compute the mean of the full and character accuracy for training
        mean_train_acc = sum(train_acc)/len(train_acc)
        mean_train_char_acc = sum(train_char_acc)/len(train_char_acc)
        train_loss = train_loss/B_size

        accuracy_train.append(mean_train_acc)
        char_accuracy_train.append(mean_train_char_acc)
        total_train_loss.append(train_loss)

        j=0
        #Validation phase
        model.eval()
        with torch.no_grad():
            for images, labels in val_dataloader:
                #print(f"Batch {j + 1}/{len(val_dataloader)}")
                images = [img.to(device) for img in images]
                labels = [lab.to(device) for lab in labels]

                images = torch.stack(images)         
                labels = torch.stack(labels)

                flat_labels_list = labels.view(-1)  
                 
                output_logits = model(images)                      
                output_probabilities = F.log_softmax(output_logits, dim=2)
                
               
                predictions = torch.argmax(output_probabilities, dim=2)
                predictions= predictions.transpose(0, 1)
                final_predictions = []
               
                for prediction in predictions:
                    reduced = []
                    before = 0
                    for t_index in prediction:
                        t_index = t_index.item()
                        if t_index != 0 and t_index != before:
                            reduced.append(t_index)
                        before = t_index
                    final_predictions.append(reduced)

                #computing the metrics for the plates
                for pred_idx_list, label in zip(final_predictions, labels):
                    label_list = label.tolist()
                    #checking if the two list are equal
                    if pred_idx_list == label_list:
                        val_acc.append(1)
                    else:
                        val_acc.append(0)
                    char_correct = 0
                    #number of same characters
                    for pred_idx, label_idx in zip(pred_idx_list, label):
                        if pred_idx == label_idx:
                            char_correct += 1
                    mean_char = char_correct/PLATE_LENGTH
                    val_char_acc.append(mean_char)
                j+=1

        #compute the mean of the iou validation score
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
    plt.savefig(f"images/train_validation_CNNCTC-{SAVE_NAME}.png")

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
    plt.savefig(f"images/char_train_validation_CNNCTC-{SAVE_NAME}.png")

    #getting the last iou value for train and validation
    final_train_acc = accuracy_train[-1]
    final_val_acc = accuracy_val[-1]

    final_char_train_acc = char_accuracy_train[-1]
    final_char_val_acc = char_accuracy_val[-1]

    with open(f"results/CNNCTC-{SAVE_NAME}.txt", "w") as f:
        f.write(f"Final train accuracy: {final_train_acc:.4f}\n")
        f.write(f"Final validation accuracy: {final_val_acc:.4f}\n")
        f.write(f"Final character train accuracy: {final_char_train_acc:.4f}\n")
        f.write(f"Final character validation accuracy: {final_char_val_acc:.4f}\n")

    #TESTING PHASE
    model.load_state_dict(torch.load(f"models/CNNCTC-{SAVE_NAME}.pth"))

    model.eval()
    test_acc = []
    char_test_acc = []
    
    #here we just loop throught the test data and compute the Iou score
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = [img.to(device) for img in images]
            labels = [lab.to(device) for lab in labels]
            images = torch.stack(images)         
            labels = torch.stack(labels)
            flat_labels_list = labels.view(-1)  
             
            output_logits = model(images)                      
            output_probabilities = F.log_softmax(output_logits, dim=2)
            
           
            predictions = torch.argmax(output_probabilities, dim=2)
            predictions= predictions.transpose(0, 1)
            final_predictions = []
            for prediction in predictions:
                    reduced = []
                    before = 0
                    for t_index in prediction:
                        t_index = t_index.item()
                        if t_index != 0 and t_index != before:
                            reduced.append(t_index)
                        before = t_index
                    final_predictions.append(reduced)

            #computing the metrics for the plates
            for pred_idx_list, label in zip(final_predictions, labels):
                label_list = label.tolist()
                #checking if the two list are equal
                if pred_idx_list == label_list:
                    test_acc.append(1)
                else:
                    test_acc.append(0)
                char_correct = 0
                #number of same characters
                for pred_idx, label_idx in zip(pred_idx_list, label):
                    if pred_idx == label_idx:
                        char_correct += 1
                mean_char = char_correct/PLATE_LENGTH
                char_test_acc.append(mean_char)               
        
    mean_acc = sum(test_acc) / len(test_acc)
    mean_char_acc = sum(char_test_acc)/len(char_test_acc)
    print(f"Test result accuracy: {mean_acc:.4f}")
    print(f"Test result char accuracy: {mean_char_acc:.4f}")

    #saving the iou result of the training, validation (last step) and testing
    with open(f"results/CNNCTC-test-{SAVE_NAME}.txt", "w") as f:
        f.write(f"Final testing accuracy: {mean_acc:.4f}\n")
        f.write(f"Final testing character accuracy: {mean_char_acc:.4f}\n")
  
