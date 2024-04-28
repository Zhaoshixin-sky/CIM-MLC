from functools import reduce
from operator import mul
from utils.util import *
from utils.graph import *

'''
generate the global buffer address for supported operators    
'''
class AddrGenerator:
    def __init__(self, onnx_model, CIM, initializer):
        self.onnx_model = onnx_model
        self.CIM = CIM
        self.initializer = initializer
        self.OpwithWeight = ['Conv', 'MatMul', 'Gemm']
        self.GBSize = CIM.GBBuf[0]
        self.BufSrc, self.BufDst = 0, 0

    def AddrGen(self):
        for name, attr in self.initializer.items():# Iterate through initializer items
            NotWeight = 1
            # Check if item is input to a weighted operator
            for node in self.onnx_model.graph.node:
                if name in node.input and node.op_type in self.OpwithWeight:
                    NotWeight = 0
            # Allocate buffer address for non-weight items
            if NotWeight:
                self.initializer[name]['addr'] = self.BufDst
                self.BufDst += reduce(mul, attr['shape'])
                self.BufSrc += reduce(mul, attr['shape'])

        for node_id, node in enumerate(self.onnx_model.graph.node):
            # get the input opeartor information
            OpSrc = GetInputOP(self.onnx_model, node_id)
            # get the op type
            OPtype = GetOPtype(node)
            # generate address for supported operators
            if OPtype == 'Conv':
                self.handle_conv(node, OpSrc)
            elif OPtype == 'Linear':
                self.handle_linear(node, OpSrc)
            elif OPtype == 'Pool':
                self.handle_pool(node, OpSrc)
            elif OPtype == 'ActivFunc':
                self.handle_activation_function(node, OpSrc)
            elif OPtype == 'ALUop':
                self.handle_alu_op(node, OpSrc)
            elif OPtype == 'Concat':
                self.handle_concat(node, OpSrc)
            elif OPtype == 'Softmax':
                self.handle_softmax(node, OpSrc)
            else:
                self.handle_other(node, OpSrc)
    
    def handle_conv(self, node, OpSrc):
        IfmSize = get_attribute(node, 'ifmsize')
        OfmSize = get_attribute(node, 'ofmsize')
        # If no input nodes, BufDst is added to the product of IfmSize (except for the first element, which is usually batch size)
        if len(OpSrc) == 0:
            self.BufDst += reduce(mul, IfmSize[1:])
        # If input nodes exist, use the storage address of the previous node as the fetch address of the current node.
        else:
            self.BufSrc = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1]

        memoryaddr = onnx.helper.make_attribute("memoryaddr", [self.BufSrc, self.BufDst])
        node.attribute.insert(-1, memoryaddr)

        self.BufDst = self.BufDst + reduce(mul, OfmSize[1:])
        # bound check
        self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

    # Gemm, matmul
    def handle_linear(self, node, OpSrc):
        IfmSize = get_attribute(node, 'ifmsize')
        OfmSize = get_attribute(node, 'ofmsize')
        if len(OpSrc) == 0:
            self.BufDst += reduce(mul, IfmSize[1:])
        else:
            self.BufSrc = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1]

        memoryaddr = onnx.helper.make_attribute("memoryaddr", [self.BufSrc, self.BufDst])
        node.attribute.insert(-1, memoryaddr)

        self.BufDst = self.BufDst + reduce(mul, OfmSize[1:])
        self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

    # pool
    def handle_pool(self, node, OpSrc):
        if not OpSrc:
            print("Pooling node has no input sources. Skipping node.")
            return
        memoryaddr_attr = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")
        # Check if memoryaddr_attr is not None
        if memoryaddr_attr is not None:
            self.BufSrc = memoryaddr_attr[-1]
        memoryaddr = onnx.helper.make_attribute("memoryaddr", [self.BufSrc, self.BufSrc])
        node.attribute.insert(-1, memoryaddr)

    # activation function
    def handle_activation_function(self, node, OpSrc):
        OfmSize = get_attribute(node, 'ofmsize')
        BufSrc = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1]

        memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc, self.BufDst])
        node.attribute.insert(-1, memoryaddr)

        self.BufDst = self.BufDst + reduce(mul, OfmSize[1:])
        self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

    # ALU ops like add, sub
    def handle_alu_op(self, node, OpSrc):
        OfmSize = get_attribute(node, 'ofmsize')
        if len(OpSrc) == 2:
            BufSrc1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1]
            BufSrc2 = get_attribute(self.onnx_model.graph.node[OpSrc[1]], "memoryaddr")[-1]

            inp1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")
            inp2 = get_attribute(self.onnx_model.graph.node[OpSrc[1]], "ofmsize")
        else:
            # If alu has only one input nodes, try using one constant as the second input
            BufSrc1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1]
            inp1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")
            init_name = get_attribute(node, 'srcinit')
            if init_name:
                BufSrc2 = self.initializer[init_name.decode("utf-8")]['addr']
                inp2 = self.initializer[init_name.decode("utf-8")]['shape']
            else:
                BufSrc2 = None
                inp2 = None

        if not inp1:
            memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc2, self.BufDst])
            node.attribute.insert(-1, memoryaddr)
        elif not inp2:
            memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc1, self.BufDst])
            node.attribute.insert(-1, memoryaddr)
        else:
            memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc1, BufSrc2, self.BufDst])
            node.attribute.insert(-1, memoryaddr)

        self.BufDst = self.BufDst + reduce(mul, OfmSize[1:])
        self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

    def handle_concat(self, node, OpSrc):
        OfmSize = get_attribute(node, 'ofmsize')
        # If the concatenation has two input nodes
        if len(OpSrc) == 2:
            # Retrieve target addresses for the two input nodes
            BufSrc1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1]
            BufSrc2 = get_attribute(self.onnx_model.graph.node[OpSrc[1]], "memoryaddr")[-1]
            # Retrieve OFM shapes for the two input nodes
            inp1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")
            inp2 = get_attribute(self.onnx_model.graph.node[OpSrc[1]], "ofmsize")
        # If len(OpSrc) is not 2
        else:
            BufSrc1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1]
            inp1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")
            # Get srcinit attribute from the current node
            init_name = get_attribute(node, 'srcinit')
            # If srcinit exists, consider it as the second input node
            if init_name:
                BufSrc2 = self.initializer[init_name.decode("utf-8")]['addr']
                inp2 = self.initializer[init_name.decode("utf-8")]['shape']
            else:
                BufSrc2 = None
                inp2 = None

        if not inp1 and not inp2:
            pass
        elif not inp1:
            memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc2, self.BufDst])
            node.attribute.insert(-1, memoryaddr)
            self.BufDst = self.BufDst + reduce(mul, inp2)
        elif not inp2:
            memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc1, self.BufDst])
            node.attribute.insert(-1, memoryaddr)
            self.BufDst = self.BufDst + reduce(mul, inp1)
        else:
            memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc1, BufSrc2, self.BufDst])
            node.attribute.insert(-1, memoryaddr)
            self.BufDst = self.BufDst + reduce(mul, inp1) + reduce(mul, inp2)

        self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

    def handle_softmax(self, node, OpSrc):
        OfmSize = get_attribute(node, 'ofmsize')
        BufSrc = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1]

        memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc, self.BufDst])
        node.attribute.insert(-1, memoryaddr)

        self.BufDst = self.BufDst + reduce(mul, OfmSize[1:])
        self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

    # interface for future supported operators
    def handle_other(self, node, OpSrc):
        if len(OpSrc) >= 1:
            memoryaddr_attr = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")
            # Use memoryaddr_attr if not None, otherwise use self.BufDst
            BufSrc = memoryaddr_attr[-1] if memoryaddr_attr is not None else self.BufDst
        else:
            BufSrc = self.BufDst

        memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc, BufSrc])
        node.attribute.insert(-1, memoryaddr)
