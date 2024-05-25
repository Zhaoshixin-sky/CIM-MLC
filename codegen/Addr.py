from functools import reduce
from operator import mul
from utils.util import *
from utils.graph import *

'''
generate the global buffer address for supported operators    
'''
class AddrGenerator:
    def __init__(self, onnx_model, CIM, initializer,const_node_output):
        self.onnx_model = onnx_model
        self.CIM = CIM
        self.initializer = initializer
        self.OpwithWeight = ['Conv', 'MatMul', 'Gemm']
        self.GBSize = CIM.GBBuf[0]
        self.BufSrc, self.BufDst = 0, 0
        self.const_node_output = const_node_output

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
                self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

        for node_id,output in self.const_node_output.items():
            # Allocate buffer address for const output items
            self.const_node_output[node_id]['addr'] = self.BufDst
            self.BufDst += reduce(mul, output['shape'])
            self.BufSrc += reduce(mul, output['shape'])
            self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

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
            elif OPtype == 'Where':
                self.handle_where(node, OpSrc)
            elif OPtype == 'Reshape':
                self.handle_reshape(node, OpSrc)
            elif OPtype == 'Expand':
                self.handle_expand(node, OpSrc)
            elif OPtype == 'Gather':
                self.handle_gather(node, OpSrc)
            elif OPtype == 'Transpose':
                self.handle_transpose(node, OpSrc)
            elif OPtype == 'Slice':
                self.handle_slice(node, OpSrc)
            elif OPtype == 'Unsqueeze':
                self.handle_unsqueeze(node, OpSrc)
            elif OPtype == 'Shape' or 'Constant' or 'ConstantOfShape':
                self.handle_1D_Constant(node,node_id, OpSrc)
            elif OPtype == 'Sqrt':
                self.handle_sqrt(node, OpSrc)
            elif OPtype == 'ReduceMean':
                self.handle_reducemean(node, OpSrc)
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

        self.BufDst = self.BufDst + reduce(mul, OfmSize[:])
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

        self.BufDst = self.BufDst + reduce(mul, OfmSize[:])
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

        self.BufDst = self.BufDst + reduce(mul, OfmSize[:])
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

        if OfmSize is not None:
            self.BufDst = self.BufDst + reduce(mul, OfmSize[:])
            self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

    def handle_concat(self, node, OpSrc): # TODO
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
        BufSrc = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1] # input data

        memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc, self.BufDst])
        node.attribute.insert(-1, memoryaddr)
        if OfmSize is not None:
            self.BufDst = self.BufDst + reduce(mul, OfmSize[:])
            self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

    # interface for future supported operators
    def handle_other(self, node, OpSrc):
        if len(OpSrc) >= 1:
            memoryaddr_attr = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")
            # Use source op's BufDst as current node's BufSrc if not None, otherwise use self.BufDst
            BufSrc = memoryaddr_attr[-1] if memoryaddr_attr is not None else self.BufDst
        else:
            BufSrc = self.BufDst
        # By default, no GBuffer space is allocated
        memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc, BufSrc])
        node.attribute.insert(-1, memoryaddr)

    def handle_slice(self, node, OpSrc):
        assert len(OpSrc) >= 3 and len(OpSrc) <= 5, "Slice must contain a minimum of 3 source Ops and a maximum of 5 "
        OfmSize = get_attribute(node, 'ofmsize')
        # 3 required OpSrc : data, starts, ends
        BufAddresses = [get_attribute(self.onnx_model.graph.node[src], "memoryaddr")[-1] for src in OpSrc[:3]]
        if len(OpSrc) > 3:
            # 2 optional OpSrc : axes, steps
            BufAddresses.extend(get_attribute(self.onnx_model.graph.node[src], "memoryaddr")[-1] for src in OpSrc[3:])
        BufAddresses.append(self.BufDst)

        memoryaddr = onnx.helper.make_attribute("memoryaddr", BufAddresses)
        node.attribute.insert(-1, memoryaddr)

        if OfmSize is not None:
            self.BufDst = self.BufDst + reduce(mul, OfmSize[:])
            self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

    def handle_unsqueeze(self, node, OpSrc):
        assert len(OpSrc) == 2, "UnSqueeze must contain exactly 2 sources Ops"
        OfmSize = get_attribute(node, 'ofmsize')
        BufSrc1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1] # data
        BufSrc2 = get_attribute(self.onnx_model.graph.node[OpSrc[1]], "memoryaddr")[-1] # axes

        memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc1,BufSrc2,self.BufDst])
        node.attribute.insert(-1, memoryaddr)

        if OfmSize is not None:
            self.BufDst = self.BufDst + reduce(mul, OfmSize[:])
            self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

    def handle_1D_Constant(self, node,node_id,OpSrc):
        OfmSize = get_attribute(node, 'ofmsize')
        if node.op_type == "Constant":# Constant
            BufSrc = self.const_node_output[node_id]['addr']
            # no data movement for constant nodes, BufDst = BufSrc
            BufDst = BufSrc
            memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc, BufDst])
        else:# Shape, ConstantOfShape, etc.
            if node.input[0] == 'input':
                BufSrc = self.BufSrc #TODO
            else:
                BufSrc = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1]
            memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc, self.BufDst])
        node.attribute.insert(-1, memoryaddr)

        if OpSrc and OfmSize is not None:
            self.BufDst = self.BufDst + reduce(mul, OfmSize[:])
            self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize


    def handle_transpose(self, node, OpSrc):
        assert len(OpSrc) == 1, "Transpose must contain exactly 1 sources Ops"
        OfmSize = get_attribute(node, 'ofmsize')
        BufSrc = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1] # data

        memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc, self.BufDst])
        node.attribute.insert(-1, memoryaddr)

        if OfmSize is not None:
            self.BufDst = self.BufDst + reduce(mul, OfmSize[:])
            self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

    def handle_reducemean(self, node, OpSrc):
        assert len(OpSrc) >= 1 and len(OpSrc) <= 2, "ReduceMean must contain a minimum of 1 source Ops and a maximum of 2"
        OfmSize = get_attribute(node, 'ofmsize')
        # 1 required OpSrc : data
        BufAddresses = [get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1]]
        if len(OpSrc) == 2:
            # 1 optional OpSrc : axes
            BufAddresses.extend(get_attribute(self.onnx_model.graph.node[OpSrc[1]], "memoryaddr")[-1])
        BufAddresses.append(self.BufDst)

        memoryaddr = onnx.helper.make_attribute("memoryaddr", BufAddresses)
        node.attribute.insert(-1, memoryaddr)

        if OfmSize is not None:
            self.BufDst = self.BufDst + reduce(mul, OfmSize[:])
            self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

    def handle_sqrt(self, node, OpSrc):
        assert len(OpSrc) == 1, "Sqrt must contain exactly 1 sources Ops"
        OfmSize = get_attribute(node, 'ofmsize')
        BufSrc = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1] # data

        memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc, self.BufDst])
        node.attribute.insert(-1, memoryaddr)

        if OfmSize is not None:
            self.BufDst = self.BufDst + reduce(mul, OfmSize[:])
            self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

    def handle_gather(self, node, OpSrc):
        assert len(OpSrc) == 2, "Gather must contain exactly two sources Ops"
        OfmSize = get_attribute(node, 'ofmsize')
        BufSrc1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1] # data
        BufSrc2 = get_attribute(self.onnx_model.graph.node[OpSrc[1]], "memoryaddr")[-1] # indices

        memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc1, BufSrc2,self.BufDst])
        node.attribute.insert(-1, memoryaddr)

        if OfmSize is not None:
            self.BufDst = self.BufDst + reduce(mul, OfmSize[:])
            self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

    def handle_reshape(self, node, OpSrc):
        assert len(OpSrc) == 2, "Reshape must contain exactly two sources Ops"
        OfmSize = get_attribute(node, 'ofmsize')
        BufSrc1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1] # data
        BufSrc2 = get_attribute(self.onnx_model.graph.node[OpSrc[1]], "memoryaddr")[-1] # shape

        memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc1, BufSrc2,self.BufDst])
        node.attribute.insert(-1, memoryaddr)

        if OfmSize is not None:
            self.BufDst = self.BufDst + reduce(mul, OfmSize[:])
            self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

    def handle_expand(self, node, OpSrc):
        OfmSize = get_attribute(node, 'ofmsize')
        if len(OpSrc) == 2: # data and shape come from SrcOp
            BufSrc1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1] # data
            BufSrc2 = get_attribute(self.onnx_model.graph.node[OpSrc[1]], "memoryaddr")[-1] # shape

            memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc1, BufSrc2, self.BufDst])
            node.attribute.insert(-1, memoryaddr)
        else: # one from initializer
            BufSrc1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1]
            # Get srcinit attribute from the current node
            init_name = get_attribute(node, 'srcinit')
            # If srcinit exists, consider it as the second input node
            if init_name:
                BufSrc2 = self.initializer[init_name.decode("utf-8")]['addr']
                memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc1, BufSrc2, self.BufDst])
                node.attribute.insert(-1, memoryaddr)
        if OfmSize is not None:
            self.BufDst = self.BufDst + reduce(mul, OfmSize[:])
            self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize

    def handle_where(self, node, OpSrc):
        assert len(OpSrc) == 3, "Where must contain exactly three sources Ops"
        OfmSize = get_attribute(node, 'ofmsize')
        BufSrc1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "memoryaddr")[-1] # condition
        BufSrc2 = get_attribute(self.onnx_model.graph.node[OpSrc[1]], "memoryaddr")[-1] # data_X
        BufSrc3 = get_attribute(self.onnx_model.graph.node[OpSrc[2]], "memoryaddr")[-1] # data_Y

        memoryaddr = onnx.helper.make_attribute("memoryaddr", [BufSrc1, BufSrc2,BufSrc3,self.BufDst])
        node.attribute.insert(-1, memoryaddr)

        if OfmSize is not None:
            self.BufDst = self.BufDst + reduce(mul, OfmSize[:])
            self.BufDst = self.BufDst if self.BufDst < self.GBSize else self.BufDst % self.GBSize