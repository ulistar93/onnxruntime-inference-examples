#!/bin/bash

source ~/torch1.8/bin/activate
python run.py --input_model best-infer.onnx --output_model best.static-quant.onnx --calibrate_dataset ./psr_images/
