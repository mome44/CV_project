import os
import torch
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, RetinaNetClassificationHead
from itertools import product


#Hyperparameters combination
batch_sizes = [16, 32]
learning_rates = [0.001, 0.002]
momentums = [0.8, 0.9]
weight_decays = [0.0001, 0.0005]
epochs = [200, 300]


combinations = product(batch_sizes, learning_rates, momentums, weight_decays, epochs)

def compute_iou(box_1, box_2):
    #it is a metric that involves the intersection of the two areas
    #over the union, and returns a matching percentage
    
    #coputing the coordinate of the intersections
    x1 = max(box_1[0], box_2[0])
    y1 = max(box_1[1], box_2[1])
    x2 = min(box_1[2], box_2[2])
    y2 = min(box_1[3], box_2[3])

    interArea = max(0, x2 - x1) * max(0, y2 - y1)
    boxAArea = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    boxBArea = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)  #+1e-6 is used to avoid the division per zero
    return iou

#executing the training and testing for all the possible combinations to get the best one
for bs, lr, mom, wd, ne in combinations:

    #Hyperparameters
    BATCH_SIZE = bs
    LR = lr
    MOMENTUM = mom
    WEIGHT_DECAY = wd
    NUM_EPOCHS = ne

    SAVE_NAME = f"n_epochs_{NUM_EPOCHS}_bs_{BATCH_SIZE}_LR_{LR}_Mom_{MOMENTUM}_wd_{WEIGHT_DECAY}"


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
    dataset = PlateDataset("dataset/images", "dataset/annotations")
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    #this optimizer uses stochastic gradient descent and has in input the parameters (weights) from 
    #the pretrained model
    params = model.parameters()
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    #initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    iou_scores_val=[]
    iou_scores_train=[]

    #TRAIN LOOP we are doing fine tuning on the task of recognizing plate
    for e in range(NUM_EPOCHS):
        model.train()
        train_loss= 0.0
        val_loss = 0.0
        train_iou = []
        val_iou = []
        #does the for loop for all the items in the same batch
        for images, labels in train_dataloader:
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
                    if len(pred['scores']) > 0:
                        #we take the index of the detected plate with the best score
                        best_index = pred['scores'].argmax()
                        best_box = pred['boxes'][best_index]

                        true_box = target['boxes'].cpu()[0]

                        single_iou = compute_iou(best_box.numpy(), true_box.numpy())
                        train_iou.append(single_iou)

        #compute the mean of the iou training score
        mean_train_iou = sum(train_iou)/len(train_iou)
        iou_scores_train.append(mean_train_iou)

        #Validation phase
        model.eval()
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = [img.to(device) for img in images]
                labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

                # Calcolo della loss (solo per monitoraggio)
                loss_dict = model(images, labels)
                loss = sum(loss for loss in loss_dict.values())
                val_loss += loss.item()

                # Inference
                val_outputs = model(images)

                for pred, target in zip(val_outputs, labels):
                    #prediction is a list that contains a score from 0 to 1 that 
                    #includes the confidence of the answer, so if this score is less than 0.5
                    #we will not consider it               
                    if len(pred['scores']) > 0:
                        #we take the index of the detected plate with the best score
                        best_index = pred['scores'].argmax()
                        best_box = pred['boxes'][best_index]

                        true_box = target['boxes'].cpu()[0]

                        single_iou = compute_iou(best_box.numpy(), true_box.numpy())
                        val_iou.append(single_iou)

        #compute the mean of the iou validation score
        mean_val_iou = sum(val_iou)/len(val_iou)
        iou_scores_val.append(mean_val_iou)

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
    plt.savefig(f"imgs/train_validation-{SAVE_NAME}.png")

    model.load_state_dict(torch.load(f"models/fasterrcnn_plate_detector-{SAVE_NAME}.pth"))

    #TESTING PHASE
    model.eval()

    iou_test = []
    #here we just loop throught the test data and compute the Iou score
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = [img.to(device) for img in images]
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            outputs = model(images)

            for pred, target in zip(outputs, labels):
                if len(pred['boxes']) == 0:
                    continue

                best_index = pred['scores'].argmax()
                pred_box = pred['boxes'][best_index].cpu().numpy()
                true_box = target['boxes'][0].cpu().numpy()

                iou = compute_iou(pred_box, true_box)
                iou_test.append(iou)

    mean_iou = sum(iou_test) / len(iou_test)
    print(f"Test result IoU: {mean_iou:.4f}")

    #getting the last iou value for train and validation
    final_train_iou = iou_scores_train[-1]
    final_val_iou = iou_scores_val[-1]

    #saving the iou result of the training, validation (last step) and testing
    with open("results/fastercnn-{SAVE_NAME}.txt", "w") as f:
        f.write(f"Final train IoU: {final_train_iou:.4f}\n")
        f.write(f"Final validation   IoU: {final_val_iou:.4f}\n")
        f.write(f"Final testing   IoU: {mean_iou:.4f}\n")

