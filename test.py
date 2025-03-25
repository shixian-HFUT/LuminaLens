import pandas
import nni
import requests
import torch
import torchvision
import numpy
import scipy
import tqdm
import sklearn
import tensorboardX
import tensorflow as tf

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.rand(2,3).to("cuda"))
print(f"pandas: {tf.__version__}")
print(f"pandas: {pandas.__version__}")
print(f"nni: {nni.__version__}")
print(f"requests: {requests.__version__}")
print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"numpy: {numpy.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"tqdm: {tqdm.__version__}")
print(f"scikit_learn: {sklearn.__version__}")
print(f"tensorboardX: {tensorboardX.__version__}")