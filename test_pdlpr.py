import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from igfe import IGFE
from encoder import PDLPR_Encoder
from decoder import ParallelDecoder
from evaluator import Evaluator
from train_pdlpr import build_vocab, load_vocab
from data import CCPDDataset
import torch.nn as nn
import torch.optim as optim
import string
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


def test(model_parts, evaluator, test_loader, char_idx, idx_char, device):
    igfe, encoder, decoder = model_parts

    # keep track of metrics for plot
    test_seq_accs = []
    test_char_accs = []
    test_lev = []

    igfe.eval()
    encoder.eval()
    decoder.eval()

    evaluator = Evaluator(idx_char)
    pbar = tqdm(test_loader, desc=f"Testing")

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
                # update the decoder with the new vocabulary size but keeping the old weights
                decoder.update_vocab_size(len(char_idx))


            # Forward
            features = igfe(images)
            encoded = encoder(features)
            logits = decoder(encoded)

            evaluator.update(logits, label_strs)
            metrics = evaluator.compute()
            test_seq_accs.append(metrics["seq_accuracy"])
            test_char_accs.append(metrics["char_accuracy"])
            test_lev.append(metrics["avg_levenshtein"])


    #saving the new vocabulary
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(char_idx, f, ensure_ascii=False, indent=2)

    metrics = evaluator.compute()
    evaluator.print()
    return metrics, test_char_accs, test_seq_accs, test_lev


# Standard transform
transform = transforms.Compose([
    transforms.Resize((48, 144)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load data
dataset = CCPDDataset(base_dir="dataset", transform=transform)
_, _, test_loader = CCPDDataset.get_dataloaders(
    base_dir="./dataset",
    batch_size=16,
    transform=transform
)

if os.path.exists('vocab.json'):
    print("vocab.json found — loading...")
    char_idx, idx_char = load_vocab('vocab.json')
else: 
    print("vocab.json not found — building it from labels...")
    char_idx, idx_char = build_vocab("dataset/labels_pdlpr/test", "vocab.json")

vocab_size = len(char_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

evaluator = Evaluator(idx_char)
ctc_loss = nn.CTCLoss(blank=0)
decoder_seq_len = 18  # From ParallelDecoder
decoder = ParallelDecoder(dim=512, vocab_size=vocab_size, seq_len=decoder_seq_len).to(device).train()
encoder = PDLPR_Encoder().to(device).train()
igfe = IGFE().to(device).train()

# load pre trained model if needed
if os.path.exists( f'PLDPR/checkpoints/pdlpr_5_0.0001_16.pt'):
    print("checkpoint found. Loading state dict......")
    checkpoint = torch.load( f'PLDPR/checkpoints/pdlpr_5_0.0001_16.pt', map_location=device)
    igfe.load_state_dict(checkpoint["igfe_state_dict"])
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

else:
    print("chackpoint not found. Please train the model first")

print("Starting testing............")
test_metrics, test_char_accs, test_seq_accs, test_lev = test(
    model_parts=(igfe, encoder, decoder),
    evaluator=evaluator,
    test_loader=test_loader,
    char_idx=char_idx,
    idx_char=idx_char,
    device=device
)