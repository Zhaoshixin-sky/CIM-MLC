from functools import reduce
from operator import mul
from utils.util import *
from utils.graph import *
from .CoreWiseCodeTem import *

class CoreWiseCodegen:
    def __init__(self,onnx_model,CIM,initializer,const_node_output,outinst = False):
        self.onnx_model = onnx_model
        self.CIM = CIM
        self.initializer = initializer
        self.inst = []
        self.outinst = outinst
        self.read_size = CIM.GBBuf[1]
        self.const_node_output = const_node_output

    def run(self):
        for node_id, node in enumerate(self.onnx_model.graph.node):
            self.process_node(node, node_id)
        if self.outinst:
            self.generate_output_instructions()
        
    def process_node(self, node, node_id):
        OPtype = GetOPtype(node)
        OpSrc = GetInputOP(self.onnx_model, node_id)

        if OPtype == 'Conv':
            self.process_conv_node_cm(node)
        elif OPtype == 'Linear':
            self.process_linear_node_cm(node)
        elif OPtype == 'Pool':
            self.process_pooling_node_cm(node)
        elif OPtype == 'ActivFunc':
            self.process_activFunc_node_cm(node)
        elif OPtype == 'ALUop':
            self.process_aluOp_node_cm(node,node_id)
        elif OPtype == 'Concat':
            self.process_concat_node_cm(node,node_id)
        elif OPtype == 'Softmax':
            self.process_softmax_node_cm(node)
        elif OPtype == 'Reshape' or OPtype =='Transpose' or OPtype =='Expand' or OPtype =='Unsqueeze' or OPtype =='Flatten':
            self.process_transform_node_cm(node,node_id,OpSrc)
        elif OPtype == 'Slice' or OPtype == 'Gather':
            self.process_indexing_node_cm(node, node_id, OpSrc)
        elif OPtype == 'Where':
            self.process_where_node_cm(node,OpSrc)
        elif OPtype == 'ReduceMean':
            self.process_reducemean_node_cm(node,OpSrc)
        elif OPtype == 'Sqrt' or OPtype == 'Abs' or OPtype == 'Exp' or OPtype == 'Log ':
            self.process_element_wise_node_cm(node,OpSrc)
        else:
            pass

    def process_conv_node_cm(self,node):
        '''
        conv - ifm[1,cin,h,w] - kernel[cout,cin,k,k] - stride[s,s] - padding[p,p,p,p] - ifmReadStart[GBbuf] - OfmWriteStart[GBbuf] 
        Mov - src[GBbuf] - dst[GBbuf] - length
        '''
        IC,_,IW,OC,OH,OW,_,K,P,S,SubIfmNum,bufsrc,bufdst = GetConvNodeInfo(node)
        S,bufsrc = S[0],bufsrc[0]
        subOH = int(OH/SubIfmNum)
        full_kernel = [OC,IC,K,K]

        CoreAddr = get_attribute(node, 'corestart')
        # Traverse the subOP after each replication
        # Divide IFM into subIFMs from the H-dimension
        for sublayer_id in range(SubIfmNum):
            if sublayer_id==0 or sublayer_id == SubIfmNum-1:# for the first subOp or the last subOp
                # redundancy calculation for marginal line
                pads = get_attribute(node,'pads')
                if SubIfmNum>1:
                    subifmsizeH = (subOH+1)*S-S+K-2*pads[-1]
                else:
                    subifmsizeH = (subOH)*S-S+K-2*pads[-1] #TODO why in this way
            else:
                pads = [1,1,0,0]
                subifmsizeH = (subOH)*S-S+K-2*pads[-1]

            # thisifmsize: record size of the subIFM
            thisifmsize = get_attribute(node,'ifmsize')
            thisifmsize[2] = subifmsizeH
            # Append conv inst which contains dimension information of the subIFM
            self.inst.append(ConvTmpl.format(
                    ifmsize=thisifmsize,
                    Kernel=full_kernel,
                    Stride=get_attribute(node, "strides"),
                    Padding=pads,
                    CoreAddrRange=f"[{CoreAddr},{CoreAddr + get_attribute(node, 'core_num') - 1}]",
                    source=bufsrc,
                    destination=bufdst,
                )
            )

            CoreAddr+=get_attribute(node,'core_num')
            # Last subOp ends with an additional Mov instruction
            if sublayer_id == SubIfmNum-1:
                self.inst.append(MovTmpl.format(src=str(bufdst + OW * OC),dst=str(bufdst),length=str((subOH)*OW*OC)))
            bufsrc = bufsrc + (subifmsizeH-2) * IW * IC
            bufdst += subOH * OW * OC

    def process_linear_node_cm(self,node):
        thisifmsize = get_attribute(node,'ifmsize')
        ofmzise = get_attribute(node,'ofmsize')

        bufsrc,bufdst = get_attribute(node,'memoryaddr')
            
        CoreAddr = get_attribute(node, 'corestart')

        self.inst.append(LinearTmpl.format(ifmsize=str(thisifmsize),
                                           CoreAddrRange=str([CoreAddr, CoreAddr + get_attribute(node, 'core_num') - 1]),
                                           source=str(bufsrc), destination=str(bufdst)))

    def process_pooling_node_cm(self,node):
        thisifmsize = get_attribute(node,'ifmsize')
        OC = get_attribute(node,'ofmsize')[1]
        OH = get_attribute(node,'ofmsize')[2]
        OW = get_attribute(node,'ofmsize')[3]
        K =  get_attribute(node,'kernel_shape')
        P = get_attribute(node,'pads')
        S = get_attribute(node,'strides')

        bufsrc,bufdst = get_attribute(node,'memoryaddr')

        self.inst.append(PoolTmpl.format(
                op_type = node.op_type,
                ifmsize = str( thisifmsize),
                Kernel = str(K),
                Stride = str(S),
                Padding = str(P),
                source = str(bufsrc),
                destination = str(bufdst)
            )
        )
        self.inst.append(MovTmpl.format(src = str(bufdst),dst = str(bufsrc), length = str(reduce(mul,get_attribute(node,'ifmsize')[:]))))

    def process_activFunc_node_cm(self,node):
        thisifmsize = get_attribute(node,'ifmsize')
        ofmzise = get_attribute(node,'ofmsize')

        bufsrc,bufdst = get_attribute(node,'memoryaddr')
        self.inst.append(ActivFuncTmpl.format(
                op_type = node.op_type,
                length = str(reduce(mul,thisifmsize[:])),
                source = str(bufsrc),
                destination = str(bufdst)
            )
        )
    
    def process_aluOp_node_cm(self,node,node_id):
        bufsrc= get_attribute(node,'memoryaddr')[:-1]
        bufdst = get_attribute(node,'memoryaddr')[-1]

        if len(bufsrc) == 2:
            OpSrc = GetInputOP(self.onnx_model,node_id)
            if len(OpSrc) == 2:
                inp1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]],"ofmsize")
                inp2 = get_attribute(self.onnx_model.graph.node[OpSrc[1]],"ofmsize")
                self.inst.append(ALUopTmpl.format(
                        op_type=node.op_type,
                        input_source1=str(bufsrc[0]),
                        input_source2=str(bufsrc[1]),
                        input_length1=str(inp1[:]),
                        input_length2=str(inp2[:]),
                        destination=str(bufdst),
                    )
                )
            else:
                inp1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]],"ofmsize")
                inp2 = self.initializer[get_attribute(node,'srcinit').decode("utf-8")]['shape']
                self.inst.append(ALUopTmpl.format(
                        op_type=node.op_type,
                        input_source1=str(bufsrc[0]),
                        input_source2=str(bufsrc[1]),
                        input_length1=str(inp1[:]),
                        input_length2=str(inp2),
                        destination=str(bufdst),
                    )
                )
    def process_concat_node_cm(self,node,node_id):
        bufaddr = get_attribute(node,'memoryaddr')
        if bufaddr:
            bufsrc= get_attribute(node,'memoryaddr')[:-1]
            bufdst = get_attribute(node,'memoryaddr')[-1]
        else:
            bufsrc= []

        if len(bufsrc) == 2:
            OpSrc = GetInputOP(self.onnx_model,node_id)
            if len(OpSrc) == 2:
                inp1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]],"ofmsize")
                inp2 = get_attribute(self.onnx_model.graph.node[OpSrc[1]],"ofmsize")
                self.inst.append(MovTmpl.format(src = str(bufdst),dst = str(bufsrc[0]),length = str(reduce(mul,inp1[:]))))
                self.inst.append(MovTmpl.format(src=str(bufdst + reduce(mul, inp1[:])), dst=str(bufsrc[1]),length=str(reduce(mul, inp2[:]))))
            else:
                inp1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]],"ofmsize")
                inp2 = self.initializer[get_attribute(node,'srcinit').decode("utf-8")]['shape']
                self.inst.append(MovTmpl.format(src=str(bufdst), dst=str(bufsrc[0]), length=str(reduce(mul, inp1[:]))))
                self.inst.append(MovTmpl.format(src=str(bufdst + reduce(mul, inp1[:])), dst=str(bufsrc[1]),length=str(reduce(mul, inp2))))

    def process_softmax_node_cm(self,node):
        thisifmsize = get_attribute(node,'ifmsize')
        ofmzise = get_attribute(node,'ofmsize')

        bufsrc,bufdst = get_attribute(node,'memoryaddr')
        self.inst.append(SoftmaxTmpl.format(
                length = str(thisifmsize[:]),
                source = str(bufsrc),
                destination = str(bufdst)
            )
        )


    def process_transform_node_cm(self,node,node_id,OpSrc):
        bufsrc= get_attribute(node,'memoryaddr')[:-1]
        bufdst = get_attribute(node,'memoryaddr')[-1]
        if len(bufsrc) == 1:  # transpose or flatten
            if len(OpSrc) == 1:
                inp = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")
                ofm = get_attribute(node, "ofmsize")
                if node.op_type == "Transpose":
                    perm = get_attribute(node, "perm")
                    self.inst.append(TransposeTmpl.format(
                        input_source=str(bufsrc[0]),
                        input_length=str(reduce(mul, inp[:])),
                        ifmsize=str(inp),
                        perm=str(perm),
                        ofmsize=str(ofm),
                        destination=str(bufdst)
                        )
                    )
                elif node.op_type == "Flatten":
                    axis = get_attribute(node, "axis")
                    self.inst.append(FlattenTmpl.format(
                        input_source=str(bufsrc[0]),
                        input_length=str(reduce(mul, inp[:])),
                        ifmsize=str(inp),
                        axis=str(axis),
                        ofmsize=str(ofm),
                        destination=str(bufdst)
                    )
                    )
        elif len(bufsrc) == 2:  # reshape, expand, or unsqueeze
            OpSrc = GetInputOP(self.onnx_model, node_id)
            if len(OpSrc) == 2:
                inp = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")
                ofm = get_attribute(node, "ofmsize")
                if node.op_type == 'Reshape':  # Reshape
                    inst = ReshapeTmpl.format(
                        op_type=str(node.op_type),
                        input_source=str(bufsrc[0]),
                        input_length=str(reduce(mul, inp[:])),
                        ifmsize=str(inp),
                        ofmsize=str(ofm),
                        destination=str(bufdst)
                    )
                elif node.op_type == 'Expand':  # Expad
                    inst = ExpandTmpl.format(
                        op_type=str(node.op_type),
                        input_source=str(bufsrc[0]),
                        input_length=str(reduce(mul, inp[:])),
                        ifmsize=str(inp),
                        ofmsize=str(ofm),
                        destination=str(bufdst)
                    )
                elif node.op_type == 'Unsqueeze':  # Unsqueeze
                    axes = get_attribute(node, "axes")
                    inst = UnsqueezeTmpl.format(
                        op_type=str(node.op_type),
                        input_source=str(bufsrc[0]),
                        input_length=str(reduce(mul, inp[:])),
                        ifmsize=str(inp),
                        axes=axes,
                        ofmsize=str(ofm),
                        destination=str(bufdst)
                    )
                self.inst.append(inst)

    def process_indexing_node_cm(self, node, node_id, OpSrc):
        bufsrc = get_attribute(node, 'memoryaddr')[:-1]
        bufdst = get_attribute(node, 'memoryaddr')[-1]

        if len(bufsrc) == 3:  # Slice
            if len(OpSrc) == 2:
                inp = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")
                starts = get_attribute(node, "starts")
                ends = get_attribute(node, "ends")
                axes = get_attribute(node, "axes")
                steps = get_attribute(node, "steps")
                ofm = get_attribute(node, "ofmsize")

                # Create the default axes list if axes is None
                axes_str = str([i for i in range(len(inp))]) if axes is None else str(axes)
                steps_str = '1' if steps is None else str(steps)

                self.inst.append(SliceTmpl.format(
                    input_source=str(bufsrc[0]),
                    input_length=str(reduce(mul, inp[:])),
                    ifmsize=str(inp),
                    starts = str(starts),
                    ends = str(ends),
                    axes = axes_str,
                    steps = steps_str,
                    ofmsize=str(ofm),
                    destination=str(bufdst)
                )
                )
        else:  # Gather
            if len(OpSrc) == 2:
                inp = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")
                indices = get_attribute(node, "indices")
                axis = get_attribute(node, "axis")
                ofm = get_attribute(node, "ofmsize")
                if inp is None:
                    inp = get_attribute(node, "inp_shape")
                self.inst.append(GatherTmpl.format(
                    input_source=str(bufsrc[0]),
                    input_length=str(reduce(mul, inp[:])),
                    ifmsize=str(inp),
                    indices = str(indices),
                    axis = str(axis),
                    ofmsize=str(ofm),
                    destination=str(bufdst)
                )
                )

    def process_where_node_cm(self,node,OpSrc):
        inp1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")
        inp2 = get_attribute(self.onnx_model.graph.node[OpSrc[1]], "ofmsize")
        inp3 = get_attribute(self.onnx_model.graph.node[OpSrc[2]], "ofmsize")
        bufsrc = get_attribute(node, 'memoryaddr')[:-1]
        bufdst = get_attribute(node, 'memoryaddr')[-1]

        ofmsize = get_attribute(node,'ofmsize')

        self.inst.append(WhereTmpl.format(
                input_source1 = str(bufsrc[0]),
                input_source2 = str(bufsrc[1]),
                input_source3 = str(bufsrc[2]),
                input_length1 = str(reduce(mul, inp1[:])),
                input_length2 = str(reduce(mul, inp2[:])),
                input_length3 = str(reduce(mul, inp3[:])),
                ofmsize=str(ofmsize),
                destination = str(bufdst)
            )
        )

    def process_reducemean_node_cm(self, node, OpSrc):
        inp = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")  # data
        axes = get_attribute(node, "axes")
        ofmsize = get_attribute(node, 'ofmsize')

        memory_addresses = get_attribute(node, 'memoryaddr')
        bufsrc = memory_addresses[0]
        bufdst = memory_addresses[-1]

        self.inst.append(ReduceMeanTmpl.format(
            input_source=str(bufsrc),
            input_length=str(reduce(mul, inp[:])),
            ifmsize=str(inp),
            axes=str(axes),
            ofmsize=str(ofmsize),
            destination=str(bufdst)
        ))

    def process_element_wise_node_cm(self, node, OpSrc):
        ifmsize = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")  # data
        ofmsize = get_attribute(node, 'ofmsize')

        memory_addresses = get_attribute(node, 'memoryaddr')
        bufsrc = memory_addresses[0]
        bufdst = memory_addresses[-1]

        self.inst.append(ElementWiseTmpl.format(
            op_type=node.op_type,
            input_source=str(bufsrc),
            input_length=str(reduce(mul, ifmsize[:])),
            ifmsize=str(ifmsize),
            ofmsize=str(ofmsize),
            destination=str(bufdst)
        ))



    def generate_output_instructions(self):
        for i in self.inst:
            print(i)