import torchvision
import torch.onnx
import torch.nn as nn
from models.experimental import attempt_load
from onnxsim import simplify
import onnx
# Standard ImageNet input - 3 channels, 224x224,
# values don't matter as we care about network structure.
# But they can also be real inputs.

device=torch.device('cuda:0')
model = attempt_load("D:/1795438624/FileRecv/polygon_last_700.pt", map_location=device)
dummy_input =torch.zeros(1, 3, 384,640).to(device)
# Invoke export
torch.onnx.export(model, dummy_input, "out12.onnx",opset_version=12)


onnx_model = onnx.load("out12.onnx")  # load onnx model
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, "y-anchor-detaset.onnx")
