import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from igfe import IGFE
from encoder import PDLPR_Encoder
from decoder import ParallelDecoder
from evaluator import Evaluator
from data import CCPDDataset
import torch.nn as nn
import torch.optim as optim
import string
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

label_folder = "dataset/labels_pdlpr/train"

#function that builds the vocabulary (chinese regions)
def build_vocab_from_label_folder(label_folder, file_name, include_blank=True):
    """
    Scans all .txt files in the label folder and builds char2idx and idx2char mappings.
    
    Args:
        label_folder (str): Path to folder containing license plate label .txt files
        include_blank (bool): Whether to reserve index 0 for the CTC blank token ('-')

    Returns:
        char2idx (dict): Character to index mapping
        idx2char (dict): Index to character mapping
    """
    vocab = set()

    for filename in os.listdir(label_folder):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(label_folder, filename), "r", encoding="utf-8") as f:
            label = f.read().strip().upper()
            vocab.update(label)

    # Sort for consistency
    vocab = sorted(vocab)

    char2idx = {}
    idx2char = {}
    start_idx = 0

    if include_blank:
        char2idx["-"] = 0  # CTC blank
        idx2char[0] = "-"
        start_idx = 1

    for i, ch in enumerate(vocab, start=start_idx):
        char2idx[ch] = i
        idx2char[i] = ch

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(char2idx, f, ensure_ascii=False, indent=2)
    
    # save the vocabulary
    print(f"[vocab] Built vocabulary with {len(char2idx)} characters.")
    return char2idx, idx2char

# function to load the vocabulary
def load_vocab(path="vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        char2idx = json.load(f)
    idx2char = {int(v): k for k, v in char2idx.items()}
    return char2idx, idx2char

def update_vocab(s, char2idx, idx2char):
    for c in s:
        if c not in char2idx:
            idx = len(char2idx)
            char2idx[c] = idx
            idx2char[idx] = c
    
    return char2idx, idx2char

def plot_metrics(train_seq, val_seq, train_char, val_char, train_lev, val_lev, num_epochs):
    epochs = range(1, len(train_losses)+1)

    plt.figure()
    plt.plot(epochs, [l.detach().cpu().item() if torch.is_tensor(l) else l for l in train_seq], label="Train Seq Accuracy")
    plt.plot(epochs, [l.detach().cpu().item() if torch.is_tensor(l) else l for l in val_seq], label="Val Seq Accuracy")
    plt.title("Sequence Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"metrics_images/seq_accs_plot_{num_epochs}.png", dpi=300)

    plt.figure()
    plt.plot(epochs, [l.detach().cpu().item() if torch.is_tensor(l) else l for l in train_char], label="Train Char Accuracy")
    plt.plot(epochs, [l.detach().cpu().item() if torch.is_tensor(l) else l for l in val_char], label="Val Char Accuracy")
    plt.title("Char Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"metrics_images/char_accs_plot{num_epochs}.png", dpi=300)

    plt.figure()
    plt.plot(epochs, [l.detach().cpu().item() if torch.is_tensor(l) else l for l in train_lev], label="Train Levenshtein distance")
    plt.plot(epochs, [l.detach().cpu().item() if torch.is_tensor(l) else l for l in val_lev], label="Val Levenshtein distance")
    plt.title("Levenshtein distance over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Lev distance")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"metrics_images/levenshtein_plot{num_epochs}.png", dpi=300)

def train(model_parts, evaluator, train_loader, char2idx, idx2char, num_epochs, optimizer,device):

    igfe, encoder, decoder = model_parts
    total_loss = 0
    
    train_losses = []
    train_seq_accs = []
    train_char_accs = []
    train_lev = []
    
    for epoch in range(num_epochs):
        evaluator = Evaluator(idx2char = idx2char)
    
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            images = batch["cropped_image"].to(device)
            label_strs = batch["pdlpr_plate_string"]
    
            # update the vocabulary if unkwon character found
            unknown = set(c for s in label_strs for c in s if c not in char2idx)
            if unknown:
                char2idx, idx2char = update_vocab(unknown, char2idx, idx2char)
                # update the decoder preserving the old weights
                decoder.update_vocab_size(len(char2idx))
    
            # Encode labels
            targets = torch.tensor([char2idx[c] for s in label_strs for c in s], dtype=torch.long).to(device)
            target_lengths = torch.tensor([len(s) for s in label_strs], dtype=torch.long).to(device)
            input_lengths = torch.full((images.size(0),), 18, dtype=torch.long).to(device)
    
            # Forward pass
            features = igfe(images)
            encoded = encoder(features)
            logits = decoder(encoded)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # [T, B, C]
    
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
    
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Update metrics
            evaluator.update(logits, label_strs)
            loop.set_postfix(loss=loss.item())
    
            metrics = evaluator.compute()
    
            train_losses.append(loss)
            train_seq_accs.append(metrics["seq_accuracy"])
            train_char_accs.append(metrics["char_accuracy"])
            train_lev.append(metrics["avg_levenshtein"])
    
        #Saving the model for testing
        torch.save({
                'epoch': epoch + 1,
                'igfe_state_dict': igfe.state_dict(),
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
                'train_losses': train_losses,
                'train_seq_accs': train_seq_accs,
                'train_char_accs': train_char_accs,
                'train_lev': train_lev
            }, f'PLDPR/checkpoints/pdlpr_final.pt')
    
    
        #saving the new vocabulary
        with open("vocab.json", "w", encoding="utf-8") as f:
            json.dump(char2idx, f, ensure_ascii=False, indent=2)

        evaluator.print()
        print(f"Loss: {total_loss / len(train_loader):.4f}")

        # plot loss over epochs
        epochs = range(1, len(train_losses)+1)
        plt.figure()
        plt.plot(epochs, [l.detach().cpu().item() if torch.is_tensor(l) else l for l in train_losses], label="Train Loss")
        plt.title("Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("CTC Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("metrics_images/loss_plot.png", dpi=300)

        return train_losses, train_seq_accs, train_char_accs, train_lev
    
def validate(model_parts, evaluator, val_loader, char2idx, idx2char, device):
    igfe, encoder, decoder = model_parts

    # keep track of metrics for plot
    val_seq_accs = []
    val_char_accs = []
    val_lev = []

    igfe.eval()
    encoder.eval()
    decoder.eval()

    evaluator = Evaluator(idx2char)
    pbar = tqdm(val_loader, desc=f"Validating")

    with torch.no_grad():
        for batch in pbar:
            images = batch["cropped_image"].to(device)
            label_strs = batch["pdlpr_plate_string"]

            unknown = set(c for s in label_strs for c in s if c not in char2idx)
            if unknown:
                char2idx, idx2char = update_vocab(unknown, char2idx, idx2char)
                decoder.update_vocab_size(len(char2idx))

            # Forward
            features = igfe(images)
            encoded = encoder(features)
            logits = decoder(encoded)


            evaluator.update(logits, label_strs)
            metrics = evaluator.compute()
            val_seq_accs.append(metrics["seq_accuracy"])
            val_char_accs.append(metrics["char_accuracy"])
            val_lev.append(metrics["avg_levenshtein"])



    metrics = evaluator.compute()
    evaluator.print()
    return metrics, val_char_accs, val_seq_accs, val_lev

if __name__ == "__main__":
    # Standard transform
    transform = transforms.Compose([
        transforms.Resize((48, 144)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    # Load data
    dataset = CCPDDataset(base_dir="dataset", transform=transform)
    train_loader, val_loader, test_loader = CCPDDataset.get_dataloaders(
        base_dir="./dataset",
        batch_size=16,
        transform=transform
    )
    
    
    if not os.path.exists('vocab.json'):
        print("vocab.json not found — building it from labels...")
        char2idx, idx2char = build_vocab_from_label_folder("dataset/labels_pdlpr/train", "vocab.json")
    else:
        print("vocab.json found — loading...")
        char2idx, idx2char = load_vocab('vocab.json')
    
    vocab_size = len(char2idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluator = Evaluator(idx2char)
    ctc_loss = nn.CTCLoss(blank=0)
    decoder_seq_len = 18  # From ParallelDecoder
    decoder = ParallelDecoder(dim=512, vocab_size=vocab_size, seq_len=decoder_seq_len).to(device).train()
    encoder = PDLPR_Encoder().to(device).train()
    igfe = IGFE().to(device).train()
    
    # TRAINING
    params = list(igfe.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=1e-4)
    num_epochs = 1 #NUMBER OF EPOCHS MODIFY HERE****************************************
        
    train_losses = []
    train_seq_accs = []
    train_char_accs = []
    train_lev = []
    
    if not os.path.exists(f'PLDPR/checkpoints/pdlpr_final.pt'):
        print("checkpoint not found. Starting training......")
        train_char_accs, train_seq_accs, train_lev, train_losses = train(
            model_parts=(igfe, encoder, decoder),
            evaluator=evaluator,
            train_loader=train_loader,
            char2idx=char2idx,
            idx2char=idx2char,
            num_epochs=num_epochs,
            optimizer=optimizer,
            device=device
        )
    
    else:
        # load pre trained model if needed
        print("checkpoint found. Loading state dict......")
        checkpoint = torch.load( f'PLDPR/checkpoints/pdlpr_final.pt', map_location=device)
        igfe.load_state_dict(checkpoint["igfe_state_dict"])
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_losses = checkpoint["train_losses"]
        train_seq_accs = checkpoint["train_seq_accs"]
        train_char_accs = checkpoint["train_char_accs"]
        train_lev = checkpoint["train_lev"]
    
    
    
    #VALIDATION
    print("Starting validation............")
    val_metrics, val_char_accs, val_seq_accs, val_lev = validate(
        model_parts=(igfe, encoder, decoder),
        evaluator=evaluator,
        val_loader=val_loader,
        char2idx=char2idx,
        idx2char=idx2char,
        device=device
    )
    
    
    print(f"[Validation] Seq Acc: {val_metrics['seq_accuracy']:.4f} | Char Acc: {val_metrics['char_accuracy']:.4f}")
    
    #plot metrics
    print("Plotting metrics.........")
    plot_metrics(train_seq_accs, val_seq_accs, train_char_accs, val_char_accs, train_lev, val_lev, num_epochs)
    