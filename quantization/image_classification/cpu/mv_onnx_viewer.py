#!/home/ycm/torch1.8/bin/python

import onnx
import onnx.numpy_helper as numpy_helper

import sys
#import pdb

if len(sys.argv) < 2:
    print("[error] this.py require [onnx_file] !")
    exit(1)

onnx_file = sys.argv[1]

model = onnx.load(onnx_file)
layers_name = list()
layers_dic = dict()
graph = model.graph
for layer in graph.initializer:
    layers_name.append(layer.name)
    layers_dic[layer.name] = numpy_helper.to_array(layer)

print("    #### input ####")
print(graph.input)
print("    #### output ####")
print(graph.output)

layers_name.sort()

for name in layers_name:
    layer = layers_dic[name]
    print("name:", name)
    print("shape type:", layer.shape, layer.dtype)
    print("    ", layer)

#pdb.set_trace()
