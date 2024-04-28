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
        'Concat':'Concat',
        'Constant': 'Constant',
        'pad':'pad'
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
        'Constant':0
        
    }
    if node.op_type in input_num:
        return input_num[node.op_type]
    else:
        return None

def GetInputOP(onnx_model,node_id):
    node = onnx_model.graph.node[node_id]
    InputNum = GetInputNum(node)
    SrcTensor = []
    src = get_attribute(node, 'srctensor')
    if src is None or len(src) == 0:
    # if len(src) == 0:
        return SrcTensor
    visited = set()
    queue = deque([node_id])
    while queue:
        tmp_node_id = queue.popleft()
        tmp_node = onnx_model.graph.node[tmp_node_id]
        visited.add(tmp_node_id)
        src = get_attribute(tmp_node, 'srctensor')
        if src is None or len(src) == 0:
            return SrcTensor
        for i in src:
            adjacent_node = onnx_model.graph.node[i]
            support = GetOPtype(adjacent_node)
            if support:
                SrcTensor.append(i)
                if len(SrcTensor)==InputNum:
                    return SrcTensor
            elif i not in visited:
                queue.append(i)
                visited.add(i)
    return SrcTensor

def GetFMSize(onnx_model,ort_outs,ifm_size):
    for node_id,node in enumerate(onnx_model.graph.node):
        # get input & output feature map size
        ofmsize = ort_outs[node.output[0]].shape
        if len(ofmsize)>=2:
            attrofm = onnx.helper.make_attribute("ofmsize",ofmsize)
            node.attribute.insert(-1,attrofm)
        # Record input dimensions for each node and add them as attributes in ifmsize format
        for inp in node.input:
            # Take the input name of the node, which is the output of another node, and treat the data as a feature map.
            # In ONNX, a Conv node has 3 input attributes, only the first is a feature map tensor (see https://onnx.ai/onnx/operators/onnx__Conv.html#l-onnx-doc-conv)
            if inp in ort_outs.keys():
                ifmsize = ort_outs[inp].shape
                if len(ifmsize)>=2:
                    attrifm = onnx.helper.make_attribute("ifmsize",ifmsize)
                    node.attribute.insert(-1,attrifm)
            # If inp is "input", it's the first node, so use the passed ifmsize as the input feature map
            elif inp=='input':
                ifmsize = ifm_size
                attrifm = onnx.helper.make_attribute("ifmsize",ifmsize)
                node.attribute.insert(-1,attrifm)

def GetDependency(onnx_model,initializer):
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

def PreProcess(onnx_model,ort_outs,ifm_size):
    # Extract initializers and constant tensors(in model.graph.initializers) from the model
    initializer = extract_initializer_value(onnx_model)
    # Add feature map size attributes to nodes
    GetFMSize(onnx_model,ort_outs,ifm_size)
    # Add dependency attributes to nodes
    GetDependency(onnx_model,initializer)
    return initializer