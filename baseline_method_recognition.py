import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import product
from data import RecognitionDataset
from network import CNN_CTC_model
#Hyperparameters combination
batch_sizes = [16, 32]
learning_rates = [0.001, 0.002]

weight_decays = [0.0001, 0.0005]
epochs = [10, 20]


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
    
    model = CNN_CTC_model(7)
    ctc_loss = nn.CTCLoss(blank=0) 
    train_dataset = RecognitionDataset("dataset/crops/train", "dataset/labels_pdlpr/train")
    val_dataset = RecognitionDataset("dataset/crops/val", "dataset/labels_pdlpr/val")
    test_dataset = RecognitionDataset("dataset/crops/test", "dataset/labels_pdlpr/test")


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    #this optimizer uses stochastic gradient descent and has in input the parameters (weights) from 
    #the pretrained model
    params = model.parameters()
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)

    #initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    accuracy_val=[]
    accuracy_train=[]

    epsilon = 1e-6
    #TRAIN LOOP we are doing fine tuning on the task of recognizing plate
    for e in range(NUM_EPOCHS):
        model.train()
        train_loss= 0.0
        val_loss = 0.0
        train_acc = []
        val_acc = []

        i=0
        #does the for loop for all the items in the same batch
        for images, labels in train_dataloader:
            print(f"Batch {i + 1}/{len(train_dataloader)}")
            images = images.to(device)                   # [B, 1, 48, 144]
            labels = labels.to(device)                   # [B, 7]
            labels_flat = labels.view(-1)                # [B×7]

            # Forward
            logits = model(images)                       # [T, B, C]
            log_probs = F.log_softmax(logits, dim=2)
            T = logits.size(0)
            B = logits.size(1)

            input_lengths = torch.full((B,), T, dtype=torch.long).to(device)
            target_lengths = torch.full((B,), 7, dtype=torch.long).to(device)

            # Loss
            loss = ctc_loss(log_probs, labels_flat, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Greedy decoding per targa intera
            preds = torch.argmax(log_probs, dim=2).transpose(0, 1)  # [B, T]
            decoded_predictions = []
            for seq in preds:
                decoded = []
                prev = 0
                for idx in seq:
                    idx = idx.item()
                    if idx != 0 and idx != prev:
                        decoded.append(idx)
                    prev = idx
                decoded_predictions.append(decoded)

            # Valutazione targa intera
            for pred_seq, label in zip(decoded_predictions, labels):
                if pred_seq == label.tolist():
                    train_acc.append(1)
                else:
                    train_acc.append(0)                    
            i+=1
            
        #compute the mean of the iou training score
        mean_train_acc = sum(train_acc)/len(train_acc)
        accuracy_train.append(mean_train_acc)

        j=0
        #Validation phase
        model.eval()
        with torch.no_grad():
            for images, labels in val_dataloader:
                print(f"Batch {j + 1}/{len(val_dataloader)}")
                images = images.to(device)
                labels = labels.to(device)
                labels_flat = labels.view(-1)

                logits = model(images)
                log_probs = F.log_softmax(logits, dim=2)
                T = logits.size(0)
                B = logits.size(1)

                input_lengths = torch.full((B,), T, dtype=torch.long).to(device)
                target_lengths = torch.full((B,), 7, dtype=torch.long).to(device)

                loss = ctc_loss(log_probs, labels_flat, input_lengths, target_lengths)
                total_val_loss += loss.item()

                preds = torch.argmax(log_probs, dim=2).transpose(0, 1)  # [B, T]
                decoded_predictions = []
                for seq in preds:
                    decoded = []
                    prev = 0
                    for idx in seq:
                        idx = idx.item()
                        if idx != 0 and idx != prev:
                            decoded.append(idx)
                        prev = idx
                    decoded_predictions.append(decoded)
                for pred_seq, label_tensor in zip(decoded_predictions, labels):
                    if pred_seq == label_tensor.tolist():
                        val_acc.append(1)
                    else:
                        val_acc.append(0)
                j+=1

        #compute the mean of the iou validation score
        mean_val_acc = sum(val_acc)/len(val_acc)
        accuracy_val.append(mean_val_acc)
        
        print(f"Epoch {e +1}/{NUM_EPOCHS} - train loss: {train_loss/len(train_dataloader):.4f} - val acc: {mean_val_acc}" )

    #Saving the model
    torch.save(model.state_dict(), f"models/next_method-{SAVE_NAME}.pth")

    #Plotting the figure for the train and validation
    plt.figure(figsize=(8, 5))
    plt.plot(NUM_EPOCHS, accuracy_train, label="train acc", marker='o')
    plt.plot(NUM_EPOCHS, accuracy_val, label="validation acc", marker='s')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Train and validation iou per epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"imgs/train_validation_next_method-{SAVE_NAME}.png")

    #getting the last iou value for train and validation
    final_train_acc = accuracy_train[-1]
    final_val_acc = accuracy_val[-1]

    with open(f"results/next_method-{SAVE_NAME}.txt", "w") as f:
        f.write(f"Final train accuracy: {final_train_acc:.4f}\n")
        f.write(f"Final validation accuracy: {final_val_acc:.4f}\n")

        

    #TESTING PHASE

    model.load_state_dict(torch.load(f"models/next_method-{SAVE_NAME}.pth"))

    model.eval()
    acc_test = []
    
    #here we just loop throught the test data and compute the Iou score
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)  # [T, B, C]
            log_probs = F.log_softmax(logits, dim=2)
            preds = torch.argmax(log_probs, dim=2).transpose(0, 1)  # [B, T]

            decoded_predictions = []
            for seq in preds:
                decoded = []
                prev = 0
                for idx in seq:
                    idx = idx.item()
                    if idx != 0 and idx != prev:
                        decoded.append(idx)
                    prev = idx
                decoded_predictions.append(decoded)

            for pred_seq, label_tensor in zip(decoded_predictions, labels):
                if pred_seq == label_tensor.tolist():
                    acc_test.append(1)
                else:
                    acc_test.append(0)                
        
    mean_acc = sum(acc_test) / len(acc_test)
    print(f"Test result accuracy: {mean_acc:.4f}")

    #saving the iou result of the training, validation (last step) and testing
    with open(f"results/next_method-test-{SAVE_NAME}.txt", "w") as f:
        f.write(f"Final testing IoU: {mean_acc:.4f}\n")
  
