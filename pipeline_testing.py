from ultralytics import YOLO
from pathlib import Path
from globals import *
import torch
import os
from pathlib import Path
from tqdm import tqdm
from utils import *
from network import *
from torchvision import transforms
from data import CCPDDataset
from torch.utils.data import DataLoader

def custom_collate(batch):
    return batch  # ritorna una lista di dizionari

def test(model_parts, yolo_model, transform, evaluator, test_loader, char_idx, idx_char, device):
    igfe, encoder, decoder = model_parts

    igfe.eval()
    encoder.eval()
    decoder.eval()

    predicted_strings = []

    evaluator = Evaluator(idx_char)
    pbar = tqdm(test_loader, desc=f"Testing")

    with torch.no_grad():
        for batch in pbar:
            # cropping images with yolo
            sample = batch[0]
            img = sample["full_image_original"]
            label_strs = sample["pdlpr_plate_string"]

            images = []

            # Crop using YOLO
            cropped_img = crop_image_yolo(yolo_model, img)
            if cropped_img is None:
                print("No plate detected — skipping image")
                continue
            # Transform cropped image
            tensor = transform(cropped_img).unsqueeze(0).to(device)
            images.append(tensor)

            if len(images) == 0:
                continue  # skip batch if all images failed

            images = torch.cat(images, dim=0)

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

            evaluator.update(logits, [label_strs])
            pred_str = evaluator.greedy_decode(logits)
            predicted_strings.append(pred_str)
            #print(f"traget string: {label_strs},  Predicted: {pred_str}")

        metrics = evaluator.compute()
        evaluator.print()

    return metrics, predicted_strings



def crop_image_yolo(yolo_model, image):
    #image = Image.open(image_path).convert("RGB")
    results = yolo_model(image, verbose=False)[0]  # Results object
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        print("No car plate detected")
        return None

    bbox = boxes.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = bbox

    cropped_img = image.crop((x1, y1, x2, y2))

    #plt.imshow(cropped_img)
    #plt.title("Targa rilevata (crop YOLO)")
    #plt.axis("off")
    #plt.show()

    return cropped_img


if __name__ == "__main__":
    # loading yolo model
    yolo_model = YOLO("runs/train/yolov5_epochs20_bs8_lr0.001_imgs640/weights/best.pt")
    
    # loading pdlpr model
    # loading vocabulary
    if os.path.exists('vocab.json'):
        print("vocab.json found — loading...")
        char2idx, idx2char = load_vocab('vocab.json')
    else: 
        print("vocab.json not found — building it from labels...")
        char2idx, idx2char = build_vocab("dataset/labels_pdlpr/train", "vocab.json")
    
    vocab_size = len(char2idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluator = Evaluator(idx2char)
    decoder_seq_len = 18  # From ParallelDecoder
    decoder = ParallelDecoder(dim=512, vocab_size=vocab_size, seq_len=decoder_seq_len).to(device).train()
    encoder = PDLPR_Encoder().to(device).train()
    igfe = IGFE().to(device).train()
    
    # load pre trained model 
    if os.path.exists( f'models/pdlpr_5_0.0001_16.pt'):
        print("checkpoint found. Loading state dict......")
        checkpoint = torch.load( f'models/pdlpr_5_0.0001_16.pt', map_location=device)
        igfe.load_state_dict(checkpoint["igfe_state_dict"])
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    else:
        print("chackpoint not found. Please train the model first")
    
    # STARTING TESTING
    # applying transformations to the image
    transform = transforms.Compose([
        transforms.Resize((48, 144)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    # crop images with YOLO
    base_dir = Path("dataset")
    image_paths = Path("dataset/images/test")
    yolo_crop_images = []

    
    # Load data
    dataset = CCPDDataset(base_dir="dataset", transform=transform)
    _, _, test_loader = CCPDDataset.get_dataloaders(
        base_dir="./dataset",
        batch_size=1,
        transform=transform
    )

    # Override del test loader con collate_fn personalizzato
    test_dataset = CCPDDataset(base_dir="dataset", transform=transform).get_dataset("test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    #pass to pdlpr
    
    #getting test labels
    label_dir = Path("dataset/labels_pdlpr/test")
    label_strs = []
    
    for txt_file in sorted(label_dir.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            label = f.read().strip()
            label_strs.append(label)
    
    print("First 5 labels:", label_strs[:5])
    
    print("start testing.........")
    metrics, predicted_strings = test(
        model_parts=(igfe, encoder, decoder),
        yolo_model = yolo_model,
        transform=transform,
        evaluator=evaluator,
        test_loader=test_loader,
        char_idx=char2idx,
        idx_char=idx2char,
        device=device
    )

    print("First 5 predicted strings:", predicted_strings[:5])