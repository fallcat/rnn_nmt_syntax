import torch

SOS_token = 0
EOS_token = 1
UNK_token = 2

MAX_LENGTH = 50
SPAN_SIZE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")