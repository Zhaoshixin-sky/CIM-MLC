import numpy as np
import einops 
import onnx
import torch
import math
from onnx import numpy_helper
from pulp import *
from utils.util import *

class RowWisePipeline:
    def __init__(self, onnx_model, CIM):
        self.onnx_model = onnx_model
        self.CIM = CIM
        self.xbsize = CIM.core[0].xb[0].Size

    def quantize(self, w):
        scale = abs(w).max() / (2**7 - 1)
        wq = np.round(w / scale)
        return wq

    def process_row_wise_pipeline(self):
        # Get initializers from ONNX model
        INTIALIZERS = self.onnx_model.graph.initializer
        conv_number = 0 # Convolution layer counter

        # Iterate over each node in the ONNX graph
        for node in self.onnx_model.graph.node:
            for initializer in INTIALIZERS:
                w = numpy_helper.to_array(initializer)
                # Check if initializer.name is in node inputs and if initializer is a 4D tensor (conv layer weights)
                if initializer.name in node.input and len(w.shape) == 4:
                    # Reshape and transpose weight array for further processing
                    w_array = w.reshape(w.shape[0], -1).transpose(1, 0)
                    wq = torch.from_numpy(self.quantize(w_array))# Quantize weights

                    # Split quantized weights into smaller chunks
                    subw = list(wq.chunk(math.ceil(wq.shape[1] / self.xbsize[1]), 1))
                    subw = [i.chunk(math.ceil(wq.shape[0] / self.xbsize[0]), 0) for i in subw]

                    # Get crossbar size-related parameters
                    rows = self.CIM.core[0].xb[0].MaxR  # Max rows that can be enabled in a crossbar
                    # Calculate physical row and column sizes occupied by a logical crossbar
                    R = self.xbsize[0] // rows
                    C = self.xbsize[1] // rows

                    # Calculate rows and columns occupied by weights
                    weight_size = w_array.shape
                    W_R = math.ceil(weight_size[0] / rows)
                    W_C = math.ceil(weight_size[1] / rows)

                    # Get custom attribute "xb_num" for the node
                    xb_num = get_attribute(node, "xb_num")
                    if xb_num is None:
                        continue

                    # Check if remapping is required based on the convolution layer's index
                    if conv_number % 2 == 0:  # Remap for even-indexed conv layers
                        RemapRes = self.get_remap_result(W_R, W_C, R, C, xb_num)
                    else:
                        # Manually create remapping result
                        RemapRes = np.arange(min(xb_num[1] * C, W_C) * min(xb_num[0] * R, W_R))
                        RemapRes = RemapRes.reshape(min(xb_num[1] * C, W_C), min(xb_num[0] * R, W_R))
                        RemapRes = np.pad(RemapRes,
                                          ((0, max(xb_num[1] * C - W_C, 0)), (0, max(xb_num[0] * R - W_R, 0))),
                                          'constant', constant_values=-1)
                        RemapRes = RemapRes.reshape(-1)
                        RemapRes = einops.rearrange(RemapRes, "(a x b y)->b a y x", a=xb_num[1], b=xb_num[0], x=C, y=R)

                    # Store remapping result
                    self.store_remap_result(node, RemapRes, W_R, W_C)
                    conv_number = conv_number + 1  # Increment convolution layer counter

    def get_remap_result(self, W_R, W_C, R, C, xb_num):
        # Initialize a list to store the remapping result of the weight matrix
        weight_record = []

        # Iterate through the weight matrix's row blocks
        for k in range(W_R):
            weight_record.append([])

            # Iterate through the weight matrix's rows
            for i in range(R):
                # Initialize a list for each row's columns
                weight_record[k].append([])

                # Iterate through the weight matrix's column blocks
                for j in range(W_C // R):
                    # Calculate and append the index of the weight in the matrix
                    if k + i * W_R + j * W_R * R <= W_R * W_C:
                        weight_record[k][i].append(k + i * W_R + j * W_R * R)
                    else:
                        weight_record[k][i].append(-1)

        # use linear indexing if W_C // R is less than 1
        if W_C // R < 1:
            weight_record = np.arange(W_R * W_C).reshape(W_C, -1).transpose(1, 0)

        # Convert the weight record to a NumPy array and flatten
        weight_record = np.array(weight_record).reshape(len(weight_record), -1)

        # Initialize the RemapRes matrix to store the mapping result
        RemapRes = np.ones((xb_num[0], xb_num[1], R, C)) * -1

        # Initialize index variables
        start = 0
        begin = [0] * len(weight_record)

        # Iterate through the dimensions of RemapRes
        for i in range(R):
            for j in range(xb_num[0]):
                for k in range(xb_num[1]):
                    # Fill RemapRes based on the index in weight_record
                    if begin[start] < len(weight_record[start]):
                        RemapRes[j, k, i, :min(C, len(weight_record[start]) - begin[start])] = \
                            weight_record[start][begin[start]:min(begin[start] + C, len(weight_record[start]))]

                    # Update the index variables
                    begin[start] = begin[start] + C
                    start = (start + 1) % W_R

        return RemapRes

    def store_remap_result(self, node, RemapRes, W_R, W_C):
        RemapRes = np.array(RemapRes, dtype=int)
        reorder_tensor = onnx.TensorProto(
            dims=RemapRes.shape,
            name="reorder_tensor",
            data_type=onnx.TensorProto.INT64,
            raw_data=RemapRes.tobytes(),
        )
        AttrRorder = onnx.helper.make_attribute("weight_reorder", reorder_tensor)
        node.attribute.extend([AttrRorder])
        weight_block = onnx.helper.make_attribute("weight_block", [W_R, W_C])
        node.attribute.extend([weight_block])

