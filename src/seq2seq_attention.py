# +
from torchtext import data,datasets
from torchtext.vocab import Vectors
import os
import torchsnooper
import random
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from apex import amp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

