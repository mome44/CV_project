from ultralytics import YOLO
from pathlib import Path
from globals import BATCH_SIZE_TRAIN_Y, LR_INIT_Y, EPOCHS_TRAIN_Y, IMAGE_SIZE_Y, IOU_THRESHOLD
#from utils import  index_to_target
from PIL import Image
import torch
import os
from pathlib import Path
from PIL import Image       # used to read image size
from tqdm import tqdm
import matplotlib.pyplot as plt
from train_pdlpr import load_vocab, build_vocab
from igfe import IGFE
from encoder import PDLPR_Encoder
from decoder import ParallelDecoder
from evaluator import Evaluator
import torch.nn as nn
from torchvision import transforms

def test_pdlpr(model_parts, evaluator, image, label_str, char_idx, idx_char, device):
    igfe, encoder, decoder = model_parts

    igfe.eval()
    encoder.eval()
    decoder.eval()

    evaluator = Evaluator(idx2char)

    image = image.unsqueeze(0).to(device)  # add batch dimension: [1, 3, 48, 144]

    # Check for unknown characters and update vocab if needed
    unknown = set(c for c in label_str if c not in char2idx)
    if unknown:
        # updating vocabulary
        for c in ''.join(label_str):
            if c not in char_idx:
                idx = len(char_idx)
                char_idx[c] = idx
                idx_char[idx] = c
        print(f"unknown character {c}. Vocabulary updated")
        # update the decoder with the new vocabulary size but keeping the old weights
        decoder.update_vocab_size(len(char_idx))

    with torch.no_grad():
        features = igfe(image)
        encoded = encoder(features)
        logits = decoder(encoded)

        evaluator.update(logits, [label_str])  # wrap label in list

    metrics = evaluator.compute()
    return {
        "logits": logits,
        "seq_accuracy": metrics["seq_accuracy"],
        "char_accuracy": metrics["char_accuracy"]
    }


def crop_image_yolo(yolo_model, image_path):
    image = Image.open(image_path).convert("RGB")
    yolo_results = yolo_model(image)[0]
    detection = yolo_results.xyxy[0]

    # Se non ci sono targhe rilevate
    if detection.shape[0] == 0:
        return []

    x1, y1, x2, y2 = detection[0].tolist()
    x1, y1, x2, y2 = map(int, detection)
    
    cropped_img = image.crop((x1, y1, x2, y2))

    plt.imshow(cropped_img)
    plt.title("Targa rilevata (crop YOLO)")
    plt.axis("off")
    plt.show()

    return cropped_img
    
    ##code to convert to plate tensor
    #with torch.no_grad():
    #    logits = pdlpr_model(plate_tensor)
    #    output_probabilities = F.log_softmax(logits, dim=2)
    #    predictions = torch.argmax(output_probabilities, dim=2)
    #    pred_text = index_to_target(logits)
    #
    #return predicted_plate

#def baseline_pipeline_prediction(cnnctc_model, image_path):
#    return predicted_plate

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
if os.path.exists( f'PLDPR/checkpoints/pdlpr_final.pt'):
    print("checkpoint found. Loading state dict......")
    checkpoint = torch.load( f'PLDPR/checkpoints/pdlpr_final.pt', map_location=device)
    igfe.load_state_dict(checkpoint["igfe_state_dict"])
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

else:
    print("chackpoint not found. Please train the model first")

# applying transformations to the image
transform = transforms.Compose([
    transforms.Resize((48, 144)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# STARTING TESTING
# applying transformations to the image
transform = transforms.Compose([
    transforms.Resize((48, 144)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# crop images with YOLO
base_dir = Path("dataset")
image_paths = Path("dataset/images/train")
yolo_crop_images = []

for image_path in image_paths.glob("*.jpg"):
    print(f"Found image: {image_path}")
    print(f"Processing: {image_path.name}")
    cropped_image = crop_image_yolo(yolo_model, image_path)
    #apply transformation to the image for pdlpr
    cropped_image = transform(cropped_image).unsqueeze(0).to(device)
    yolo_crop_images.append(cropped_image)

print("all images cropped using yolo!")

#pass to pdlpr

#print("Starting testing pdlpr............")
#test_metrics, test_char_accs, test_seq_accs, test_lev = test(
#    model_parts=(igfe, encoder, decoder),
#    evaluator=evaluator,
#    image=input_tensor,
#    label_str=label_str,
#    char2idx=char2idx,
#    idx2char=idx2char,
#    device=device
#)

