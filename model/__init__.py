import torch

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3


MAX_LENGTH = 50
SPAN_SIZE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DEVICES = torch.cuda.device_count() if torch.cuda.is_available() else 1