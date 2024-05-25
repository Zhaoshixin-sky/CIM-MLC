import torch
import math
import copy
import onnx
import onnxruntime
import numpy as np
from .graph import TreeNode
from collections import OrderedDict,deque
from onnx import helper

# get attribute in onnx model
def get_attribute(node, attr_name, default_value=None):
    found = [attr for attr in node.attribute if attr.name == attr_name]
    if found:
        return helper.get_attribute_value(found[0])
    return default_value

def ModelParser(model,ifg):
    ofm = []
    ifm = []
    def farward_hook(module, inp, outp):
        ofm.append(list(outp.shape))
        ifm.append(list(inp[0].shape))
    for name, m in model.named_modules():
        if type(m) in [torch.nn.Conv2d, torch.nn.MaxPool2d]:
            m.register_forward_hook(farward_hook)
    outs = model(ifg)
    index = 0
    layer = {}
    for name, m in model.named_modules():
        if type(m) in [torch.nn.Conv2d]:          
            layer[name] = ['Conv2d',[ifm[index][1:],ofm[index][1:],m]]
            index+=1
        elif type(m) in [torch.nn.MaxPool2d]:
            layer[name] = ['MaxPool2d',[ifm[index][1:],ofm[index][1:],m]]
            index+=1
    GraphNodeSet =[TreeNode('root','root',[])]
    for key in layer.keys():
        GraphNodeSet.append(TreeNode(tag=key,type=layer[key][0],param=layer[key][1]))
    return GraphNodeSet


def get_layer_output(model, image):
    # Backup original model outputs for later restoration
    ori_output = copy.deepcopy(model.graph.output)

    # Loop through each node in the model, save output names, and add to model.graph.output
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    # Create an ONNX Runtime session for inference
    ort_session = onnxruntime.InferenceSession(model.SerializeToString())

    # Prepare model inputs
    ort_inputs = {}
    for i, input_ele in enumerate(ort_session.get_inputs()):
        # Set the same image data for each input element (assuming identical data for each input)
        ort_inputs[input_ele.name] = image

    # Get names of all output layers in the model
    outputs = [x.name for x in ort_session.get_outputs()]

    # Run the model and get outputs for each layer
    ort_outs = ort_session.run(outputs, ort_inputs)

    # Pack output names and data into an ordered dictionary
    ort_outs = OrderedDict(zip(outputs, ort_outs))

    # Clear current model outputs (model.graph.output list)
    del model.graph.output[:]

    # Restore original model outputs (backed up at the beginning)
    model.graph.output.extend(ori_output)

    # Return a dictionary of layer outputs, with output names as keys and data as values
    return ort_outs


def extract_constant_value(constant_node):
    # Get the attribute representing the constant value
    attr = constant_node.attribute[-1]

    # Convert the raw data to a numpy array based on the data type
    if attr.type == onnx.TensorProto.FLOAT:
        np_value = np.frombuffer(attr.t.raw_data, dtype=np.float32)
    elif attr.type == onnx.TensorProto.INT32:
        np_value = np.frombuffer(attr.t.raw_data, dtype=np.int32)
    elif attr.type == onnx.TensorProto.INT64:
        np_value = np.frombuffer(attr.t.raw_data, dtype=np.int64)
    elif attr.type == onnx.TensorProto.STRING:
        np_value = attr.t.string_data
    else:
        np_value = None  # Unsupported data type

    return np_value

def extract_initializer_value(onnx_model):
    initializer_node = {}
    for initializer in onnx_model.graph.initializer:
        # Get the data type and shape of the initializer tensor
        data_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[initializer.data_type]
        shape = list(initializer.dims)
        # Get the value of the initializer tensor
        value = initializer.raw_data
        # Convert the raw_data to a numpy array based on the data type
        np_value = np.frombuffer(value, dtype=data_type)
        initializer_node[initializer.name] = {'data_type':data_type,'shape':shape,'value':np_value}
    return initializer_node

def Greedy(GraphNodeSet, r_tmp):
    for i in range(len(GraphNodeSet)):
        print(i,GraphNodeSet[i].tag)
    while r_tmp>0:
        l_set = [Node.latency for Node in GraphNodeSet]
        n_set = []
        for Node in GraphNodeSet:
            if Node.latency==max(l_set):
                n_set.append(Node)
        for Node in n_set:
            r_tmp-=Node.base_resource
        if r_tmp<0:break

        for Node in n_set:
            Node.resource+=Node.base_resource
            Node.latency=Node.base_latency/(Node.resource/Node.base_resource)

def GetConvNodeInfo(node):
    xb_num = get_attribute(node,'xb_num')
    Dup = get_attribute(node,'dup')
    K =  get_attribute(node,'kernel_shape')[0]
    P = get_attribute(node,'pads')
    S = get_attribute(node,'strides')
    IC = get_attribute(node,'ifmsize')[1]
    IH = get_attribute(node,'ifmsize')[2]
    IW = get_attribute(node,'ifmsize')[3]
    OC = get_attribute(node,'ofmsize')[1]
    OH = get_attribute(node,'ofmsize')[2]
    OW = get_attribute(node,'ofmsize')[3]
    bufsrc = get_attribute(node,'memoryaddr')[:-1]
    bufdst = get_attribute(node,'memoryaddr')[-1]    
    return IC,IH,IW,OC,OH,OW,xb_num,K,P,S,Dup,bufsrc,bufdst


def GetOPtype(node):
    op_type = {
        'Conv':'Conv',
        'MaxPool':'Pool',
        'AveragePool':'Pool',
        'Relu':'ActivFunc',
        'Sigmoid':'ActivFunc',
        'Tanh':'ActivFunc',
        'Erf': 'ActivFunc',
        'Softmax':'Softmax',
        'Sqrt':'Sqrt',
        'ReduceMean':'ReduceMean',
        'GlobalAveragePool':'GlobalPool',
        'MatMul':'Linear',
        'Gemm':'Linear',
        'Add':'ALUop',
        'Mul':'ALUop',
        'Sub':'ALUop',
        'Pow':'ALUop',
        'Div':'ALUop',
        'Equal':'ALUop',#  Let's do it this way for now.
        'Concat':'Concat',
        'Constant': 'Constant',
        'ConstantOfShape': 'ConstantOfShape',
        'Shape': 'Shape',
        'pad':'pad',
        'Slice':'Slice',
        'Expand':'Expand',
        'Gather':'Gather',
        'Transpose':'Transpose',
        'Reshape':'Reshape',
        'Unsqueeze':'Unsqueeze',
        'Flatten':'Flatten',
        'Where':'Where' #  do it this way for now.
    }
    if node.op_type in op_type:
        return op_type[node.op_type]
    else:
        return None

def GetInputNum(node):
    # Input count for each node type
    input_num = {
        'Conv':1,
        'MaxPool':1,
        'AveragePool':1,
        'Relu':1,
        'Sigmoid':1,
        'Tanh':1,
        'Softmax':1,
        'Sqrt':1,
        'ReduceMean':1,
        'GlobalAveragePool':1,
        'MatMul':1,
        'Gemm':1,
        'Add':2,
        'Mul':2,
        'Sub':2,
        'Pow':2,
        'Div':2,
        'Concat':2,
        'pad':2,
        'Constant':0,
        'Where':3,
        'Reshape':2,
        'Equal':2,
        'Shape':1,
        'ConstantOfShape':1,
        'Gather':2,
        'Expand':2,
        'Slice':5,
        'Transpose':1,
        'Unsqueeze':2,
        'Flatten':1
    }
    if node.op_type in input_num:
        return input_num[node.op_type]
    else:
        return None

def GetInputOP(onnx_model, node_id):
    # Get the node from the ONNX model by its ID.
    node = onnx_model.graph.node[node_id]
    # Determine the number of inputs expected for this node.
    InputNum = GetInputNum(node)
    # Initialize a list to hold the source tensor IDs.
    SrcTensor = []
    # Attempt to get the 'srctensor' attribute from the node.
    src = get_attribute(node, 'srctensor')
    # If there is no 'srctensor' attribute or it is empty, return an empty list.
    if src is None or len(src) == 0:
        return SrcTensor
    # Initialize a set to keep track of visited nodes and a queue for BFS traversal.
    visited = set()
    queue = deque([node_id])
    # Perform BFS traversal to find all input nodes.
    while queue:
        # Dequeue a node ID from the front of the queue.
        tmp_node_id = queue.popleft()
        # Get the corresponding node from the model.
        tmp_node = onnx_model.graph.node[tmp_node_id]
        # Mark the node as visited.
        visited.add(tmp_node_id)
        # Get the 'srctensor' attribute of the current node.
        src = get_attribute(tmp_node, 'srctensor')
        # If there is no 'srctensor' attribute or it is empty, return the current SrcTensor list.
        if src is None or len(src) == 0:
            return SrcTensor
        # Iterate over each source tensor ID in the 'srctensor' attribute.
        for i in src:
            # Get the adjacent node from the model.
            adjacent_node = onnx_model.graph.node[i]
            # Check if the operation type of the adjacent node is supported.
            support = GetOPtype(adjacent_node)
            # If the operation type is supported, add the node ID to SrcTensor.
            if support:
                SrcTensor.append(i)
                # If we have found all expected input nodes, return SrcTensor.
                if len(SrcTensor) == InputNum:
                    return SrcTensor
            # If the node has not been visited and is not a supported operation, enqueue it for further traversal.
            elif i not in visited:
                queue.append(i)
                visited.add(i)
    # Return the SrcTensor list with all found input node IDs.
    return SrcTensor


# def GetFMSize(onnx_model,ort_outs,ifm_size):
#     for node_id,node in enumerate(onnx_model.graph.node):
#         # get input & output feature map size
#         ofmsize = ort_outs[node.output[0]].shape
#         if len(ofmsize)>=2:# Don't add the ofmsize attribute if  ofmsize of the node is 1-dimensional or 0-dimensional,
#             attrofm = onnx.helper.make_attribute("ofmsize",ofmsize)
#             node.attribute.insert(-1,attrofm)
#         # Record input dimensions for each node and add them as attributes in ifmsize format
#         for inp in node.input:
#             # Take the input name of the node, which is the output of another node, and treat the data as a feature map.
#             # In ONNX, usually, if a node has multiple inputs, only the first is a input feature map tensor (see https://onnx.ai/onnx/operators/onnx__Conv.html#l-onnx-doc-conv)
#             if inp in ort_outs.keys():
#                 ifmsize = ort_outs[inp].shape
#                 if len(ifmsize)>=2:# Don't add the ifmsize attribute if ifmsize of the node is 1-dimensional or 0-dimensional
#                     attrifm = onnx.helper.make_attribute("ifmsize",ifmsize)
#                     node.attribute.insert(-1,attrifm)
#             # If inp is "input", it means that the node is the first node, so the passed ifmsize (function parameter) is used
#             elif inp=='input':
#                 ifmsize = ifm_size
#                 attrifm = onnx.helper.make_attribute("ifmsize",ifmsize)
#                 node.attribute.insert(-1,attrifm)

def GetFMSize(onnx_model,ort_outs,ifm_size):
    for node_id,node in enumerate(onnx_model.graph.node):
        # get input & output feature map size
        ofmsize,ofm_data_num = ort_outs[node.output[0]].shape,ort_outs[node.output[0]].size
        if len(ofmsize) > 0:# Don't add the ofmsize attribute if  ofmsize of the node is 0-dimensional,
            attrofm = onnx.helper.make_attribute("ofmsize",ofmsize)
            node.attribute.insert(-1,attrofm)
        elif len(ofmsize)==0 and ofm_data_num > 0:
            attrofm = onnx.helper.make_attribute("ofmsize", [ofm_data_num])
            node.attribute.insert(-1, attrofm)

        # Record input dimensions for each node and add them as attributes in ifmsize format
        for inp in node.input:
            # Take the input name of the node, which is the output of another node, and treat the data as a feature map.
            # In ONNX, usually, if a node has multiple inputs, only the first is a input feature map tensor (see https://onnx.ai/onnx/operators/onnx__Conv.html#l-onnx-doc-conv)
            if inp in ort_outs.keys():
                ifmsize,ifm_data_num = ort_outs[inp].shape,ort_outs[inp].size
                if len(ifmsize) > 0:# Don't add the ifmsize attribute if ifmsize of the node is 0-dimensional
                    attrifm = onnx.helper.make_attribute("ifmsize",ifmsize)
                    node.attribute.insert(-1,attrifm)
                elif len(ifmsize) == 0 and ifm_data_num > 0:
                    attrifm = onnx.helper.make_attribute("ifmsize", [ifm_data_num])
                    node.attribute.insert(-1, attrifm)
            # If inp is "input", it means that the node is the first node, so the passed ifmsize (function parameter) is used
            elif inp=='input':
                ifmsize = ifm_size
                attrifm = onnx.helper.make_attribute("ifmsize",ifmsize)
                node.attribute.insert(-1,attrifm)

def GetDependency(onnx_model,initializer,ort_outs):
    input_record = []
    output_record = []
    connection = []
    for node_id,node in enumerate(onnx_model.graph.node):
        # get the dependency relation of operators
        input_record.append(node.input)
        output_record.append(node.output)
        connection.append([])

        # Get the input name of the current node
        inp_name = input_record[node_id]
        # Check if current node's inputs depend on previous nodes' outputs
        for i in range(len(output_record)):
            for j in output_record[i]:
                if j in inp_name:
                    connection[node_id].append(i)
        # If there are dependencies, add an attribute to the node
        if connection[node_id]:
            attrsrc = onnx.helper.make_attribute("srctensor", connection[node_id])
            node.attribute.insert(-1, attrsrc)

        # Get standard number of inputs for the current node
        InpNum = GetInputNum(node)
        # If the node has fewer dependencies than expected inputs
        if InpNum and len(connection[node_id])<InpNum:
            # Check if any inputs are initializers (e.g. features.0.weight, in the initializer list)
            for name in node.input:
                if name in initializer:
                    # If input is an initializer, add "srcinit" attribute to the node
                    attrinitsrc = onnx.helper.make_attribute("srcinit",name)
                    node.attribute.insert(-1,attrinitsrc)

def GetSpecAttrs(onnx_model,ort_outs):
    for node_id, node in enumerate(onnx_model.graph.node):
        if node.op_type == 'Gather':
            indices = ort_outs[node.input[1]]
            if len(indices.shape) == 0:
            # and len(ort_outs[node.input[0]].shape) == 1:
                indices = [0]
            # print(indices)
            # print(node.input)
            # print([ort_outs[node.input[0]],ort_outs[node.input[1]]])
            attr_indices = onnx.helper.make_attribute("indices", indices)
            node.attribute.insert(-1, attr_indices)
        if node.op_type == 'Slice':
            starts = ort_outs[node.input[1]]
            ends = ort_outs[node.input[2]]
            attr_starts = onnx.helper.make_attribute("starts", starts)
            attr_ends = onnx.helper.make_attribute("ends", ends)
            node.attribute.append(attr_starts)
            node.attribute.append(attr_ends)

            if len(node.input) > 3:
                axes = ort_outs[node.input[3]]
                attr_axes = onnx.helper.make_attribute("axes", axes)
                node.attribute.append(attr_axes)
            if len(node.input) > 4:
                steps = ort_outs[node.input[4]]
                attr_steps = onnx.helper.make_attribute("steps", steps)
                node.attribute.append(attr_steps)
        if node.op_type == 'Gather':
            for inp in node.input:
                if inp in ort_outs.keys() and inp == 'output':
                    inp_shape = ort_outs[inp].shape
                    attr_inp = onnx.helper.make_attribute("inp_shape", inp_shape)
                    node.attribute.append(attr_inp)
        if node.op_type == 'Unsqueeze':
            for inp in node.input:
                if inp in ort_outs.keys():
                    axes = ort_outs[inp]
                    axes = axes.reshape(1) if len(axes.shape)==0 else axes
                    attr_axes = onnx.helper.make_attribute("axes", axes)
                    node.attribute.append(attr_axes)

def GetConstOutput(onnx_model,ort_outs):
    const_node_output = {}
    for node_id, node in enumerate(onnx_model.graph.node):
        if node.op_type == 'Constant':
            shape_attr = get_attribute(node,'ofmsize')
            v_attr = get_attribute(node,'value')
            data_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[v_attr.data_type]
            value = v_attr.raw_data
            # Convert the raw_data to a numpy array based on the data type
            np_value = np.frombuffer(value, dtype=data_type)
            const_node_output[node_id] = {'name':node.output[0], 'data_type':data_type,'shape': shape_attr, 'value': np_value}
    return const_node_output
# def GetShape(onnx_model,initializer,ort_outs):
#     input_record = []
#     output_record = []
#     connection = []
#
#
#     for node_id,node in enumerate(onnx_model.graph.node):
#         if node.op_type == 'Reshape':
#             OpSrc = GetInputOP(onnx_model, node_id)
#             if len(node.input) == 2:
#                 ifm_shape = ort_outs[node.input][0]
#                 ofm_shape = ort_outs[node.input][-1]
#                 if len(ofm_shape) == 1 and ofm_shape[0] == -1:
#
#
#             # self.initializer[get_attribute(node, 'srcinit').decode("utf-8")]['shape']
#             # get the dependency relation of operators
#             input_record.append(node.input)
#             output_record.append(node.output)
#             connection.append([])
#
#             # Get the input name of the current node
#             inp_name = input_record[node_id]
#             # Check if current node's inputs depend on previous nodes' outputs
#             for i in range(len(output_record)):
#                 for j in output_record[i]:
#                     if j in inp_name:
#                         connection[node_id].append(i)
#             # If there are dependencies, add an attribute to the node
#             if connection[node_id]:
#                 attrsrc = onnx.helper.make_attribute("srctensor", connection[node_id])
#                 node.attribute.insert(-1, attrsrc)
#
#             # Get standard number of inputs for the current node
#             InpNum = GetInputNum(node)
#             # If the node has fewer dependencies than expected inputs
#             if InpNum and len(connection[node_id])<InpNum:
#                 # Check if any inputs are initializers (e.g. features.0.weight, in the initializer list)
#                 for name in node.input:
#                     if name in initializer:
#                         # If input is an initializer, add "srcinit" attribute to the node
#                         attrinitsrc = onnx.helper.make_attribute("srcinit",name)
#                         node.attribute.insert(-1,attrinitsrc)

def PreProcess(onnx_model,ort_outs,ifm_size):
    # Extract initializers and constant tensors(in model.graph.initializers) from the model
    initializer = extract_initializer_value(onnx_model)
    # Add feature map size attributes to nodes
    GetFMSize(onnx_model,ort_outs,ifm_size)
    # Add dependency attributes to nodes
    GetDependency(onnx_model,initializer,ort_outs)
    # Add perm attribute to reshape nodes
    GetSpecAttrs(onnx_model,ort_outs)
    # extract outputs value of Constant nodes  (in ort_orts)
    const_node_output = GetConstOutput(onnx_model,ort_outs)
    return initializer,const_node_output


def image_to_input_ids(image_data):
    """
    Convert image data to a serialized form of input_ids.

    Parameters:
        image_data (np.ndarray): A batch of images with shape [batch, num_channels, height, width]
                                 and dtype float.

    Returns:
        input_ids (np.ndarray): A batch of images with shape [batch, seq_length]
                                 and dtype int64.
    """
    input_ids = (image_data * 255).astype(np.int64)
    input_ids = input_ids.reshape(input_ids.shape[0], -1)  # Flatten all dimensions except the batch dimension

    return input_ids

def image_to_attention_mask(image_data):
    """
    Convert image data to a serialized form of attention_mask.

    Parameters:
        image_data (np.ndarray): A batch of images with shape [batch, num_channels, height, width]
                                 and dtype float.

    Returns:
        attention_mask (np.ndarray): A batch of images with shape [batch, seq_length]
                                 and dtype int64.
    """
    attention_mask = np.ones_like(image_data, dtype=np.int64)
    attention_mask = attention_mask.reshape(attention_mask.shape[0], -1)  # Flatten all dimensions except the batch dimension

    return attention_mask
