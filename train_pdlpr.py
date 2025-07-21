import json
import torch
import torchvision.transforms as transforms
import os
from network import *
from utils import *
from globals import *
from data import CCPDDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt

label_folder = "dataset/labels_pdlpr/train"  


#function that builds the vocabulary (chinese regions)

def train(model_parts, evaluator, train_loader, val_loader, char_idx, idx_char, num_epochs, optimizer ,device):

    igfe, encoder, decoder = model_parts
    total_loss = 0
    
    train_losses = []
    train_seq_accs = []
    train_char_accs = []

    val_char_accs = []
    val_seq_accs = [] 
    
    for epoch in range(num_epochs):
        evaluator = Evaluator(idx2char=idx_char)
    
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            images = batch["cropped_image"].to(device)
            label_strs = batch["pdlpr_plate_string"]
    
            # update the vocabulary if unkwon character found
            unknown = set(c for s in label_strs for c in s if c not in char_idx)
            if unknown:
                # updating vocabulary
                for c in ''.join(label_strs):
                    if c not in char_idx:
                        idx = len(char_idx)
                        char_idx[c] = idx
                        idx_char[idx] = c
                print(f"unknown character {c}. Vocabulary updated")
                # update the decoder with the new vocabulary size but keeping the old weights
                decoder.update_vocab_size(len(char_idx))
    
            # lable encoding in ordr to compute loss
            targets = torch.tensor([char_idx[c] for s in label_strs for c in s], dtype=torch.long).to(device)
            target_lengths = torch.tensor([len(s) for s in label_strs], dtype=torch.long).to(device)
            input_lengths = torch.full((images.size(0),), 18, dtype=torch.long).to(device)
    
            # Forward pass
            features = igfe(images)
            encoded = encoder(features)
            logits = decoder(encoded)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # [T, B, C]
    
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            train_losses.append(loss)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # metrics updating using the evaluator
            evaluator.update(logits, label_strs)
            loop.set_postfix(loss=loss.item())
    
        print(f"Metrics at epoch {epoch+1}")
        evaluator.print()
        metrics = evaluator.compute()
        train_seq_accs.append(metrics["seq_accuracy"])
        train_char_accs.append(metrics["char_accuracy"])
    
    
        #saving the new vocabulary
        with open("vocab.json", "w", encoding="utf-8") as f:
            json.dump(char_idx, f, ensure_ascii=False, indent=2)

        # VALIDATION
        val_evaluator = Evaluator(idx_char)
        val_metrics = validate(
        model_parts=(igfe, encoder, decoder),
        evaluator=val_evaluator,
        val_loader=val_loader,
        char_idx=char_idx,
        idx_char=idx_char,
        device=device
    )
        val_seq_accs.append(val_metrics["seq_accuracy"])
        val_char_accs.append(val_metrics["char_accuracy"])
    
    #Saving the model for testing, the models will have as input the images cropped by YOLO 
    torch.save({
            'epoch': epoch + 1,
            'igfe_state_dict': igfe.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'loss': total_loss,
            'train_losses': train_losses,
            'train_seq_accs': train_seq_accs,
            'train_char_accs': train_char_accs
        }, f'models/pdlpr_{NUM_EPOCHS_PDLPR}_{LR_PDLPR}_{BATCH_SIZE_PDLPR}.pt')
    print("Model saved in models/pdlpr_final.pt")

    print("END OF TRAINING, results:\n")

    print(f"number of epochs: {num_epochs}")
    print(f"learning rate: {LR_PDLPR}")
    print(f"batch size: {BATCH_SIZE_PDLPR}")
    print(f"Loss: {total_loss / len(train_loader):.4f}")
    evaluator.print()


    train_seq_accuracy = metrics['seq_accuracy']
    val_seq_accuracy = val_metrics["seq_accuracy"]
    train_char_accuracy = metrics['char_accuracy']
    val_char_accuracy = metrics['char_accuracy']

    with open(f"results/PDLPR-{NUM_EPOCHS_PDLPR}_{LR_PDLPR}_{BATCH_SIZE_PDLPR}.txt", "w") as f:
        f.write(f"Final train accuracy: {train_seq_accuracy:.4f}\n")
        f.write(f"Final validation accuracy: {val_seq_accuracy:.4f}\n")
        f.write(f"Final character train accuracy: {train_char_accuracy:.4f}\n")
        f.write(f"Final character validation accuracy: {val_char_accuracy:.4f}\n")

    print(f"results saved in results/PDLPR-{NUM_EPOCHS_PDLPR}_{LR_PDLPR}_{BATCH_SIZE_PDLPR}.txt")

    # plot loss over epochs
    epochs = range(1, len(train_losses)+1)
    plt.figure()
    plt.plot(epochs, [l.detach().cpu().item() if torch.is_tensor(l) else l for l in train_losses], label="Train Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("CTC Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"metrics_images/loss_plot_{num_epochs}_{LR_PDLPR}_{BATCH_SIZE_PDLPR}.png", dpi=300)

    # plot train and validation metrics
    print("Plotting metrics.........")
    plot_metrics(train_seq_accs, val_seq_accs, train_char_accs, val_char_accs)

    return train_losses, train_seq_accs, train_char_accs
    
def validate(model_parts, evaluator, val_loader, char_idx, idx_char, device):
    igfe, encoder, decoder = model_parts

    igfe.eval()
    encoder.eval()
    decoder.eval()

    evaluator = Evaluator(idx_char)
    pbar = tqdm(val_loader, desc=f"Validating")

    with torch.no_grad():
        for batch in pbar:
            images = batch["cropped_image"].to(device)
            label_strs = batch["pdlpr_plate_string"]

            unknown = set(c for s in label_strs for c in s if c not in char_idx)
            if unknown:
                # updating vocabulary
                for c in ''.join(label_strs):
                    if c not in char_idx:
                        idx = len(char_idx)
                        char_idx[c] = idx
                        idx_char[idx] = c
                print(f"unknown character {c}. Vocabulary updated")
                decoder.update_vocab_size(len(char_idx))

            # Forward
            features = igfe(images)
            encoded = encoder(features)
            logits = decoder(encoded)

            evaluator.update(logits, label_strs)

    metrics = evaluator.compute()
    evaluator.print()
    return metrics

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((48, 144)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    # loading data

    dataset = CCPDDataset(base_dir="dataset", transform=transform)
    train_loader, val_loader, test_loader = CCPDDataset.get_dataloaders(
        base_dir="./dataset",
        batch_size=BATCH_SIZE_PDLPR,
        transform=transform,
        collate_fn= custom_collate
    )
    
    
    if not os.path.exists('vocab.json'):
        print("vocab.json not found — building it from labels...")
        char_idx, idx_char = build_vocab("dataset/labels_pdlpr/train", "vocab.json")
    else:
        print("vocab.json found — loading...")
        char_idx, idx_char = load_vocab('vocab.json')
    
    vocab_size = len(char_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluator = Evaluator(idx_char)
    ctc_loss = nn.CTCLoss(blank=0)
    decoder_seq_len = 18  # From ParallelDecoder
    decoder = ParallelDecoder(dim=512, vocab_size=vocab_size, seq_len=decoder_seq_len).to(device).train()
    encoder = PDLPR_Encoder().to(device).train()
    igfe = IGFE().to(device).train()
    
    # TRAINING
    params = list(igfe.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=LR_PDLPR, weight_decay=0.0001)
    
    print("Starting training..........")
    train_char_accs, train_seq_accs, train_losses = train(
        model_parts=(igfe, encoder, decoder),
        evaluator=evaluator,
        train_loader=train_loader,
        val_loader=val_loader,
        char_idx=char_idx,
        idx_char=idx_char,
        num_epochs=NUM_EPOCHS_PDLPR,
        optimizer=optimizer,
        device=device
    )
    

        