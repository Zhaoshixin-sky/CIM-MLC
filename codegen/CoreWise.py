from functools import reduce
from operator import mul
from utils.util import *
from utils.graph import *
from .CoreWiseCodeTem import *

class CoreWiseCodegen:
    def __init__(self,onnx_model,CIM,initializer,outinst = False):
        self.onnx_model = onnx_model
        self.CIM = CIM
        self.initializer = initializer
        self.inst = []
        self.outinst = outinst
        self.read_size = CIM.GBBuf[1]

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
                    subifmsizeH = (subOH)*S-S+K-2*pads[-1]
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
            bufsrc = bufsrc+ (subifmsizeH-2)  * IW * IC
            bufdst += subOH*OW*OC

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
        self.inst.append(MovTmpl.format(src = str(bufdst),dst = str(bufsrc), length = str(reduce(mul,get_attribute(node,'ifmsize')[1:]))))

    def process_activFunc_node_cm(self,node):
        thisifmsize = get_attribute(node,'ifmsize')
        ofmzise = get_attribute(node,'ofmsize')

        bufsrc,bufdst = get_attribute(node,'memoryaddr')
        self.inst.append(ActivFuncTmpl.format(
                op_type = node.op_type,
                length = str(thisifmsize[1:]),
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
                        input_length1=str(inp1[1:]),
                        input_length2=str(inp2[1:]),
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
                        input_length1=str(inp1[1:]),
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
                self.inst.append(MovTmpl.format(src = str(bufdst),dst = str(bufsrc[0]),length = str(reduce(mul,inp1[1:]))))
                self.inst.append(MovTmpl.format(src=str(bufdst + reduce(mul, inp1[1:])), dst=str(bufsrc[1]),length=str(reduce(mul, inp2[1:]))))
            else:
                inp1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]],"ofmsize")
                inp2 = self.initializer[get_attribute(node,'srcinit').decode("utf-8")]['shape']
                self.inst.append(MovTmpl.format(src=str(bufdst), dst=str(bufsrc[0]), length=str(reduce(mul, inp1[1:]))))
                self.inst.append(MovTmpl.format(src=str(bufdst + reduce(mul, inp1[1:])), dst=str(bufsrc[1]),length=str(reduce(mul, inp2))))

    def process_softmax_node_cm(self,node):
        thisifmsize = get_attribute(node,'ifmsize')
        ofmzise = get_attribute(node,'ofmsize')

        bufsrc,bufdst = get_attribute(node,'memoryaddr')
        self.inst.append(SoftmaxTmpl.format(
                length = str(thisifmsize[1:]),
                source = str(bufsrc),
                destination = str(bufdst)
            )
        )

    def generate_output_instructions(self):
        for i in self.inst:
            print(i)