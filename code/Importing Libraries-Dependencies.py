import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import random

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm

from torch.cuda import is_available
device_gpu=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
