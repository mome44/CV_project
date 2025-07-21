DATASET_PATH_Y      = "dataset"
IMAGE_SIZE_Y        = 640       # alla fine: 640, per il testing su CPU: 416

BATCH_SIZE_TRAIN_Y  = 20
BATCH_SIZE_TEST_Y   = 4
EPOCHS_TRAIN_Y      = 20
LR_INIT_Y           = 0.001     # initial learning rate
# LR_END_Y            = 0.00001   # final learning rate
# DECAY_Y             = 0.003


BATCH_SIZE_PDLPR = 16
LR_PDLPR = 1e-4 #0.00001, mostly used
NUM_EPOCHS_PDLPR = 10  

IOU_THRESHOLD       = 0.7

#character mapping used for the chinese plates
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

CHAR_LIST = sorted(set(PROVINCES+ALPHABETS+ADS))

CHAR_IDX = {}
IDX_CHAR = {}
for idx, char in enumerate(CHAR_LIST):
    CHAR_IDX[char] = idx + 1  # start from 1
    IDX_CHAR[idx + 1] = char
IDX_CHAR[0] = '_'  # blank character for CTC
