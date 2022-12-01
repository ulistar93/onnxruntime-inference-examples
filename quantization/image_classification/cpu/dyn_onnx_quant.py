#!/home/ycm/torch1.8/bin/python

import sys

if len(sys.argv) < 2:
    print("[error] this.py require [fp32_onnx_file] !")
    exit(1)

model_fp32 = sys.argv[1]

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_name = ".onnx".join(model_fp32.split('/')[-1].split(".onnx")[:-1])
model_quant = model_name + '.quant.onnx'

print(model_fp32, "->", model_quant)
quantized_model = quantize_dynamic(model_fp32, model_quant)
