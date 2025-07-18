import os
import torch
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor 
from itertools import product
from data import PlateDataset
from utils import compute_iou
from globals import IOU_THRESHOLD
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
#Hyperparameters combinations
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
    #loading the pretrained model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    #model = retinanet_resnet50_fpn(weights="DEFAULT")

    #Since we want that the last classifier layer to have only an output of two
    #so we get the input dimension of it
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    #and reinitialize it with the new class, we have to put num classes = 2 because otherwise the model
    #will always be sure that there will be a plate
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # background + plate

    #in_channels = model.head.classification_head.conv[0].in_channels
    #num_anchors = model.head.classification_head.num_anchors
    #model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes=2)

    #cambiare il dataloader in modo che ce ne sia uno per train validation e test
    
    train_dataset = PlateDataset("dataset/images/train", "dataset/labels/train")
    val_dataset = PlateDataset("dataset/images/val", "dataset/labels/val")
    test_dataset = PlateDataset("dataset/images/test", "dataset/labels/test")


    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    #this optimizer uses stochastic gradient descent and has in input the parameters (weights) from 
    #the pretrained model
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    #initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    iou_scores_val=[]
    iou_scores_train=[]

    precision_train=0
    recall_train = 0
    f1_train =0

    precision_val = 0
    recall_val = 0
    f1_val = 0

    epsilon = 1e-6
    #TRAIN LOOP we are doing fine tuning on the task of recognizing plate
    for e in range(NUM_EPOCHS):
        model.train()
        train_loss= 0.0
        val_loss = 0.0
        train_iou = []
        val_iou = []
        #tp, fp, fn to be used in the last epoch
        TP_train = 0
        FP_train = 0
        FN_train = 0
        TP_val = 0
        FP_val = 0
        FN_val = 0

        i=0
        #does the for loop for all the items in the same batch
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {e+1}/{NUM_EPOCHS} - Training"):
            #print(f"Batch {i + 1}/{len(train_dataloader)}")
            #moves the images and labels to the GPU
            images = list(img.to(device) for img in images)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            #in this model there are different losses
            loss_dict = model(images, labels)
            #we do the sum of all of them to compute the total loss
            total_step_loss = sum(loss for loss in loss_dict.values())

            #we remove the gradients from the previous steps
            optimizer.zero_grad()
            #compute the new gradients
            total_step_loss.backward()
            #update the weights that are computed now.
            optimizer.step()

            train_loss += total_step_loss.item()
            #put the model in the evaluation phase to get the preditions
           
            model.eval() 
            with torch.no_grad():
                train_outputs = model(images)
            
            model.train()
            for pred, target in zip(train_outputs, labels):
                #prediction is a list that contains a score from 0 to 1 that 
                #includes the confidence of the answer, so if this score is less than 0.5
                #we will not consider it   
                print(pred['scores']) 
                if len(pred['scores']) > 0:
                    #we take the index of the detected plate with the best score
                    best_index = pred['scores'].argmax()
                    best_box = pred['boxes'][best_index]
                    true_box = target['boxes'].cpu()[0]
                    #print(best_box,true_box)
                    single_iou = compute_iou(best_box.numpy(), true_box.numpy())
                    #train_iou.append(single_iou)
                    
                    if single_iou > IOU_THRESHOLD:
                        train_iou.append(1)
                        TP_train += 1
                    else:
                        train_iou.append(0)
                        FP_train += 1
                else:
                    FN_train += 1
            i+=1
        print(train_iou)
        #compute the mean of the iou training score
        mean_train_iou = sum(train_iou)/len(train_iou)
        iou_scores_train.append(mean_train_iou)

        j=0
        #Validation phase
        model.eval()
        with torch.no_grad():
            for images, labels in val_dataloader:
                print(f"Batch {j + 1}/{len(val_dataloader)}")
                images = [img.to(device) for img in images]
                labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

                # Inference
                val_outputs = model(images)

                for pred, target in zip(val_outputs, labels):
                    #prediction is a list that contains a score from 0 to 1 that 
                    #includes the confidence of the answer, so if this score is less than 0.5
                    #we will not consider it
                    print(pred['scores'])              
                    if len(pred['scores']) > 0:
                        #we take the index of the detected plate with the best score
                        best_index = pred['scores'].argmax()
                        best_box = pred['boxes'][best_index]

                        true_box = target['boxes'].cpu()[0]

                        single_iou = compute_iou(best_box.numpy(), true_box.numpy())
                        
                        #val_iou.append(single_iou)
                        if single_iou > IOU_THRESHOLD:
                            val_iou.append(1)
                            TP_val += 1
                        else:
                            val_iou.append(0)
                            FP_val += 1
                    else:
                        FN_val += 1
                j+=1

        #compute the mean of the iou validation score
        mean_val_iou = sum(val_iou)/len(val_iou)
        iou_scores_val.append(mean_val_iou)
        
        if e == NUM_EPOCHS-1 :
            #in the last epoch I compute the precision recall f1 score metrics for training and validation
            precision_train = TP_train / (TP_train + FP_train + epsilon)
            recall_train = TP_train / (TP_train + FN_train + epsilon)
            f1_train = 2 * precision_train * recall_train / (precision_train + recall_train + epsilon)

            precision_val= TP_val / (TP_val + FP_val + epsilon)
            recall_val = TP_val / (TP_val + FN_val + epsilon)
            f1_val = 2 * precision_val * recall_val / (precision_val + recall_val + epsilon)

        print(f"Epoch {e +1}/{NUM_EPOCHS} - train loss: {train_loss/len(train_dataloader):.4f} - val mIou: {mean_val_iou}" )

    #Saving the model
    torch.save(model.state_dict(), f"models/fasterrcnn_plate_detector-{SAVE_NAME}.pth")

    #Plotting the figure for the train and validation
    plt.figure(figsize=(8, 5))
    plt.plot(NUM_EPOCHS, iou_scores_train, label="train IoU", marker='o')
    plt.plot(NUM_EPOCHS, iou_scores_val, label="validation IoU", marker='s')
    plt.xlabel("epoch")
    plt.ylabel("IoU")
    plt.title("Train and validation iou per epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"imgs/train_validation_fastercnn-{SAVE_NAME}.png")

    #getting the last iou value for train and validation
    final_train_iou = iou_scores_train[-1]
    final_val_iou = iou_scores_val[-1]

    with open(f"results/fastercnn-{SAVE_NAME}.txt", "w") as f:
        f.write(f"Final train IoU: {final_train_iou:.4f}\n")
        f.write(f"Final validation IoU: {final_val_iou:.4f}\n")
        f.write(f"Final train precision: {precision_train:.4f}\n")
        f.write(f"Final validation precision: {precision_val:.4f}\n")
        f.write(f"Final train recall: {recall_train:.4f}\n")
        f.write(f"Final validation recall: {recall_val:.4f}\n")
        f.write(f"Final train f1: {f1_train:.4f}\n")
        f.write(f"Final validation f1: {f1_val:.4f}\n")
        

    #TESTING PHASE

    model.load_state_dict(torch.load(f"models/fasterrcnn_plate_detector-{SAVE_NAME}.pth"))

    model.eval()
    epsilon = 1e-6
    iou_test = []
    TP_test = 0
    FP_test = 0
    FN_test = 0
    precision_test = 0
    recall_test = 0
    f1_test = 0
    #here we just loop throught the test data and compute the Iou score
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = [img.to(device) for img in images]
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            outputs = model(images)

            for pred, target in zip(outputs, labels):
                if len(pred['scores']) > 0:
                        

                    best_index = pred['scores'].argmax()
                    pred_box = pred['boxes'][best_index].cpu().numpy()
                    true_box = target['boxes'][0].cpu().numpy()

                    iou = compute_iou(pred_box, true_box)
                    if iou > IOU_THRESHOLD:
                        iou_test.append(1)
                        TP_test +=1
                    else:
                        iou_test.append(0)
                        FP_test +=1
                else:
                    FN_test += 1
        precision_test= TP_test / (TP_test + FP_test + epsilon)
        recall_test = TP_test / (TP_test + FN_test + epsilon)
        f1_test = 2 * precision_test * recall_test / (precision_test + recall_test + epsilon)

    mean_iou = sum(iou_test) / len(iou_test)
    print(f"Test result IoU: {mean_iou:.4f}, precision: {precision_test:.4f}, recall: {recall_test:.4f}, f1 score: {f1_test:.4f}")

    #saving the iou result of the testing
    with open(f"results/fastercnn-test-{SAVE_NAME}.txt", "w") as f:
        f.write(f"Final testing IoU: {mean_iou:.4f}\n")
        f.write(f"Final testing precision: {precision_test:.4f}\n")
        f.write(f"Final testing recall: {recall_test:.4f}\n")
        f.write(f"Final testing f1: {f1_test:.4f}\n")


