import torch.nn.functional as F
from functools import reduce
from operator import mul
from utils.util import *
from utils.graph import *
from .CrossbarCodeTem import *

def xb_occupation_update(xb_ocp_dict, begin, end, fetch_time):
    """
        Find and update the earliest available time for each XB.
    Args:
        xb_ocp_dict: A dictionary recording the occupation status of XBs, with XB id as keys and an array of the earliest idle moments as values.
        begin: Starting XB id.
        end: Ending XB id.
        fetch_time: The latest ready time of IFM data points.
    """
    max_cycle = 0
    # Find the earliest available time of the needed XB
    for i in range(begin, end): # Traverse all XBs from index begin to end
        if i in xb_ocp_dict:
            max_cycle = max(max_cycle, max(xb_ocp_dict[i]))

    # cycle = later time {latest ready time of IFM data points, latest ready time of XBs(from index begin to end)}
    #       = time when XBs(from index begin to end) can start working
    cycle = max(fetch_time, max_cycle)

    # Update XB occupation time
    for i in range(begin, end): # Traverse XBs from index begin to end
        if i in xb_ocp_dict:
            xb_ocp_dict[i].append(cycle + 2)
        else:
            xb_ocp_dict[i] = [cycle + 2]
    return cycle


def inst_record_update(instdict, time, node_id, inst):
    """
    Record node_id's execution of inst at time in instdict.
    Args:
        instdict:   Dict with timestamps as keys and node_id-instruction pairs as values.
        time:       Timestamp of the event.
        node_id:    Identifier of the node.
        inst:       Instruction executed by the node.
    """
    # Add time if not in instdict
    if time not in instdict:
        instdict[time] = {node_id: [inst]}  # Create new entry for time
    else:
        # Add node_id if not in instdict[time]
        if node_id not in instdict[time]:
            instdict[time][node_id] = [inst]  # Create new entry for node_id
        else:
            instdict[time][node_id].append(inst)  # Add inst to node_id's list


def conv_start_time_retrieval(onnx_model, node, OpSrc, IC, IH, IW):
    # Initialize or retrieve cycle records for operations
    if not OpSrc:  # If no previous operations
        source_data_fetch_time = [0] * IC * IH * IW  # Initialize with zeros
    else:
        # Use last OpSrc's cycle_record as current Op's fetch_time
        for i in OpSrc:
            if len(get_attribute(onnx_model.graph.node[i], 'cycle_record')) == IC * IH * IW:
                source_data_fetch_time = get_attribute(onnx_model.graph.node[i], 'cycle_record')
                break  # Assuming we only need the first valid cycle_record

    # Set start time for non-pipelined network segments
    CoreAddr = get_attribute(node, "corestart")
    # For non-sequential FlowSegs, the fetch time for the first Conv node is the maximum ready cycle of the last node in the previous FlowSeg
    if CoreAddr == 0:
        source_data_fetch_time = [max(source_data_fetch_time)] * len(source_data_fetch_time)
    return source_data_fetch_time


def mul_start_time_retrieval(onnx_model, node, OpSrc,ifmsize):
    source_data_fetch_time = []
    # Initialize or retrieve cycle records for operations
    if len(OpSrc) == 0:
        source_data_fetch_time = [0] * reduce(mul, ifmsize[:])
    else:
        for i in OpSrc:
            if len(get_attribute(onnx_model.graph.node[i], 'cycle_record')) == reduce(mul, ifmsize[:]):
                source_data_fetch_time = get_attribute(onnx_model.graph.node[i], 'cycle_record')

    # Initialization of start time for segmented network. Segmented network cannot undergo pipeline.
    CoreAddr = get_attribute(node, "corestart")
    if CoreAddr == 0 and source_data_fetch_time:
        source_data_fetch_time = [max(source_data_fetch_time)] * len(source_data_fetch_time)

    # exception
    if len(source_data_fetch_time) == 0 or len(source_data_fetch_time) != reduce(mul, ifmsize[:]):
            source_data_fetch_time = [0] * reduce(mul, ifmsize[:])

    return source_data_fetch_time

class CrossbarWiseCodegen:
    def __init__(self, onnx_model, CIM, initializer, outinst = False):
        self.onnx_model = onnx_model
        self.CIM = CIM
        self.initializer = initializer
        self.xbdst = 0
        self.total_inst = []
        self.source_data_fetch_time = []
        self.compute_cycle = {}
        self.pooling_cycle = {}
        self.mov_cycle = {}
        self.activ_cycle = {}
        self.alu_cycle = {}
        self.softmax_cycle = {}
        self.transform_cycle = {}
        self.indexing_cycle = {}
        self.where_cycle = {}
        self.reduction_cycle = {}
        self.elementwise_op_cycle = {}
        self.xb_cycle = {}
        self.outinst = outinst

    def run(self):
        for node_id, node in enumerate(self.onnx_model.graph.node):
            self.process_node(node, node_id)
        if self.outinst:
            self.generate_output_instructions()

    def process_node(self, node, node_id):
        loadinst = []
        OPtype = GetOPtype(node)
        OpSrc = GetInputOP(self.onnx_model, node_id)

        if OPtype == 'Conv':
            self.process_conv_node_xbm(node, node_id, OpSrc)
        elif OPtype == 'Linear':
            self.process_linear_node_xbm(node, node_id, OpSrc)
        elif OPtype == 'Pool':
            self.process_pooling_node_xbm(node,node_id,OpSrc)
        elif OPtype == 'ActivFunc':
            self.process_activFunc_node_xbm(node,node_id,OpSrc)
        elif OPtype == 'ALUop':
            self.process_aluOp_node_xbm(node,node_id,OpSrc)
        elif OPtype == 'Softmax':
            self.process_softmax_node_xbm(node, node_id, OpSrc)
        elif OPtype == 'Reshape' or OPtype == 'Transpose' or OPtype == 'Expand' or OPtype == 'Unsqueeze' or OPtype =='Flatten':
            self.process_transform_node_xbm(node, node_id, OpSrc)
        elif OPtype == 'Where':
            self.process_where_node_xbm(node, node_id, OpSrc)
        elif OPtype == 'ReduceMean':
            self.process_reduction_node_xbm(node, node_id, OpSrc)
        elif OPtype == 'Slice' or OPtype == 'Gather':
            self.process_indexing_node_xbm(node, node_id, OpSrc)
        elif OPtype == 'Sqrt' or OPtype == 'Abs' or OPtype == 'Exp' or OPtype == 'Log ':
            self.process_elementwise_node_xbm(node, node_id, OpSrc)
        elif OPtype == 'Shape' or 'Constant' or 'ConstantOfShape':
            self.process_constant_node_xbm(node, node_id, OpSrc)
        else:
            ofmsize = get_attribute(node, 'ofmsize')
            # Use source node's cycle_record if available
            if len(OpSrc) >= 1 and len(get_attribute(self.onnx_model.graph.node[OpSrc[0]], 'cycle_record')) > 0:
                self.source_data_fetch_time = get_attribute(self.onnx_model.graph.node[OpSrc[0]], 'cycle_record')
                # If SrcOp's ofm_size differs from current Op's, take the latest ready time from SrcOp and reshape to match current Op's ofm_size
                if ofmsize and len(self.source_data_fetch_time) != reduce(mul, ofmsize[:]):
                    self.source_data_fetch_time = [max(self.source_data_fetch_time)] * reduce(mul, ofmsize[:])
                attrcycle = onnx.helper.make_attribute("cycle_record", self.source_data_fetch_time)
                node.attribute.insert(-1, attrcycle)
            # If no source node, take the latest ready time of self.source_data_fetch_time as current Op's cycle_record
            else:
                if ofmsize and len(self.source_data_fetch_time) != reduce(mul, ofmsize[:]):
                    if len(self.source_data_fetch_time) == 0:
                        self.source_data_fetch_time = [0] * reduce(mul, ofmsize[:])
                    else:#
                        self.source_data_fetch_time = [max(self.source_data_fetch_time)] * reduce(mul, ofmsize[:])
                attrcycle = onnx.helper.make_attribute("cycle_record", self.source_data_fetch_time)
                node.attribute.insert(-1, attrcycle)

    def process_conv_node_xbm(self, node, node_id, OpSrc):
        # Extract convolution node information such as dimensions, buffer sources, etc.
        IC, IH, IW, OC, OH, OW, xb_num, K, P, S, dup, bufsrc, bufdst = GetConvNodeInfo(node)
        bufsrc = bufsrc[0]
        # Get fetch time for each input pixel without padding.
        # If OpSrc exists, use its cycle_record; otherwise, initialize to 0.
        # Shape: [IC * IH * IW]
        self.source_data_fetch_time = conv_start_time_retrieval(self.onnx_model,node,OpSrc,IC,IH,IW)

        # cycle_record: Ready time for each pixel in OFM after computation (fetch, compute, store)
        cycle_record = [0] * OC * OH * OW

        # xb id for computation
        CoreAddr = get_attribute(node, "corestart")
        # For the first node in a FlowSeg, set its target xb address to 0
        if CoreAddr == 0:
            self.xbdst = 0
        dst = self.xbdst

        # Record input load addresses, considering padding and strides
        InputVecAddr = []

        if P is None:
            P = [0, 0]

        # data layerout: (H,W,CHANNEL) for example: address 0~32, the of 32 different channel for the same point (0,0)
        TIH = IH + 2 * P[0]
        TIW = IW + 2 * P[1]
        # InputVecAddr
        # Shape: (OH * OW, K)
        # For each position (oh, ow) in OFM, InputVecAddr[i] stores K pairs of (start, end) addresses,
        # which specify the ranges of data rows in IFM that affect the computation at position (oh, ow) in OFM.
        # InputVecAddr[i]: the starting and ending positions in IFM of the data at position (i // OW, i % OW) in OFM.
        for oh in range(0, OH):
            for ow in range(0, OW):
                # start address in the input vector for the output point (oh,ow)
                buftmp = (oh * TIW * S[0] + ow * S[1]) * IC
                # the input address for all K lines. For example, record the three lines with x address
                # .....
                # .xxx.
                # .xxx.
                # .xxx.
                # .....
                tmp = []
                for kh in range(K):
                    tmp.append([buftmp, buftmp + K * IC])
                    buftmp = buftmp + TIW * IC
                # Add the input data address for (oh,ow) to InputVecAddr
                InputVecAddr.append(tmp)

        # generate code for each computation
        bias = (TIW - 2 * P[1]) * IC * P[0] + P[1] * IC
        up_pad_bound = IC * TIW * P[0]  # Values less than this indicate that the entire range is padding content
        down_pad_bound = (TIH - P[0]) * IC * TIW  # Values great than this indicate that the entire range is padding content
        left_pad_bound = P[1] * IC  # left
        right_pad_bound = TIW * IC - P[1] * IC  # right

        for ComNum, VecAddr in enumerate(InputVecAddr):
            cycle = 0
            tmpinp = []
            for v in VecAddr:
                scale = int(v[0] / (TIW * IC)) * IC * 2 * P[1]
                excess_value = bias + scale
                row_left = v[0] % (TIW * IC)  # Offset of v[0] from first num in row (incl. padding)
                row_right = v[1] % (TIW * IC)  # Offset of v[1] from first num in same row (incl. padding)
                data_fetch_time = [0]*2 # facilitate to get the ealiest computation start time

                # Up and down padding - no need for data loading, just padding
                if v[0] < up_pad_bound or v[0] >= down_pad_bound:
                    tmpinp.append(['pad-' + str(K * IC * P[0])])
                # left padding needed
                elif row_left < left_pad_bound:
                    tmpinp.append(['pad-' + str(left_pad_bound - row_left),
                                   bufsrc + v[0] + left_pad_bound - row_left - excess_value,
                                   bufsrc + v[1] - excess_value])
                    data_fetch_time.extend(self.source_data_fetch_time[v[0] + left_pad_bound - row_left - excess_value:v[1] - excess_value])
                # right padding needed
                elif row_right > right_pad_bound:
                    tmpinp.append([bufsrc + v[0] - excess_value,
                                   bufsrc + v[1] - (row_right - right_pad_bound) - excess_value,
                                   'pad-' + str(row_right - right_pad_bound)])
                    data_fetch_time.extend(self.source_data_fetch_time[v[0] - excess_value:v[1] - (v[1] % (TIW * IC) - (TIW - P[1]) * IC) - excess_value])
                elif row_right == 0:
                    tmpinp.append(
                        [bufsrc + v[0] - excess_value, bufsrc + v[1] - IC * P[1] - excess_value,
                         'pad-' + str(IC * P[1])])
                    data_fetch_time.extend(self.source_data_fetch_time[v[0] - excess_value:v[1] - IC * P[1] - excess_value])
                # no padding needed
                else:
                    tmpinp.append([bufsrc + v[0] - excess_value, bufsrc + v[1] - excess_value])
                    data_fetch_time.extend(self.source_data_fetch_time[v[0] - excess_value:v[1] - excess_value])

            # Find the latest ready time for all IFM data points covered by InputVecAddr[i]
            v_fetch_time = max(data_fetch_time)
            # cycle: the earliest idle time for target xb
            cycle = xb_occupation_update(self.xb_cycle,dst,dst+ (xb_num[0] * xb_num[1]),v_fetch_time)
            # Record the ready time for OC points in the current OFM corresponding to InputVecAddr[i]
            # Add 3 cycles (load - compute - store) to the cycle value
            cycle_record[ComNum * OC:(ComNum + 1) * OC] = [cycle+3] * OC

            # codegen
            # Create mov inst with {source}, {destination range}, {length}
            mov_inst = MovTmpl.format(source=tmpinp,destination=[dst, dst + (xb_num[0] * xb_num[1]) - 1],length=0) # TODO length
            inst_record_update(self.mov_cycle,cycle+1,node_id,mov_inst) # Schedule load instruction after 1 cycle
            # Create compute instruction with {XB address range}
            compute_inst = ConvTmpl.format(xbaddr=[dst, dst + (xb_num[0] * xb_num[1]) - 1])
            inst_record_update(self.compute_cycle,cycle+2,node_id,compute_inst) # Schedule compute instruction after 2 cycles

            # used crossbar update
            # Distribute the data across dup XBs by adjusting the dst value.
            if (ComNum + 1) % dup == 0:
                dst = self.xbdst
            else:
                dst += (xb_num[0] * xb_num[1])

        # update the output acquisition time
        # cycle_record : shape = (OH * OW * OC)
        attrcycle = onnx.helper.make_attribute("cycle_record", cycle_record)
        node.attribute.insert(-1, attrcycle)
        # Update the global starting xbdst value by adding the resources used by the current Node.
        self.xbdst += xb_num[0] * xb_num[1] * dup

    def process_linear_node_xbm(self, node, node_id, OpSrc):
        thisifmsize = get_attribute(node, 'ifmsize')
        ofmsize = get_attribute(node, 'ofmsize')

        # source & dest address
        memoryaddr = get_attribute(node, 'memoryaddr')
        # debug information
        if not memoryaddr or len(memoryaddr) < 2:
            print(
                f"Exception: 'memoryaddr' attribute is missing or incomplete in node {node.name}. Skipping this node.")
            return
        ts = memoryaddr[0]
        td = memoryaddr[1]

        self.source_data_fetch_time = mul_start_time_retrieval(self.onnx_model,node,OpSrc,thisifmsize)

        xb_num = get_attribute(node, 'xb_num')
        xb_size = self.CIM.core[0].xb[0].Size
        CoreAddr = get_attribute(node, "corestart")
        if CoreAddr == 0:
            self.xbdst = 0
        dst = self.xbdst
        cycle_record = []

        for i in range(thisifmsize[-2]):
            # load data address
            begin = i * thisifmsize[-1]
            end = (i + 1) * thisifmsize[-1]

            max_cycle = max(self.source_data_fetch_time[begin:end])

            max_cycle = xb_occupation_update(self.xb_cycle,dst,dst + (xb_num[0] * xb_num[1]),max_cycle)

            cycle_record.extend([max_cycle+3] * ofmsize[-1])

            mov_inst = MovTmpl.format(source=ts,
                               destination=[dst, dst + xb_num[0] * xb_num[1]],
                               length=thisifmsize[-1])
            inst_record_update(self.mov_cycle,max_cycle+1,node_id,mov_inst)

            mul_inst = ConvTmpl.format(xbaddr=[dst, dst + xb_num[0] * xb_num[1]])
            inst_record_update(self.compute_cycle,max_cycle+2,node_id,mul_inst)

            ts = ts + thisifmsize[-1]

        dst = dst + xb_num[0] * xb_num[1]
        self.xbdst = dst

        self.source_data_fetch_time = cycle_record
        attrcycle = onnx.helper.make_attribute("cycle_record", self.source_data_fetch_time)
        node.attribute.insert(-1, attrcycle)

    def process_pooling_node_xbm(self, node, node_id, OpSrc):
        thisifmsize = get_attribute(node, 'ifmsize')
        ofmsize = get_attribute(node, 'ofmsize')
        K = get_attribute(node, 'kernel_shape')
        P = get_attribute(node, 'pads')
        S = get_attribute(node, 'strides')
        if P is None or len(P) == 0:#  If not present, the padding defaults to 0 along start and end of each spatial axis.
            P = len(thisifmsize) * [0]
        memoryaddr = get_attribute(node, 'memoryaddr')
        if not memoryaddr or len(memoryaddr) < 2:
            print(f"Exception: 'memoryaddr' attribute is missing or incomplete in node {node.name}. Skipping this node.")
            return
        ts = memoryaddr[0]
        td = memoryaddr[1]

        self.source_data_fetch_time = get_attribute(self.onnx_model.graph.node[OpSrc[0]], 'cycle_record')

        if self.source_data_fetch_time:
            cycle_record = torch.tensor(self.source_data_fetch_time) + 3.0
            cycle_record = cycle_record.reshape(thisifmsize[0], thisifmsize[2], thisifmsize[3], thisifmsize[1]).permute(0, 3, 1, 2)
            cycle_record = F.max_pool2d(cycle_record, K, S, P[0])
            cycle_record = cycle_record.permute(0, 2, 3, 1).reshape(-1).int().tolist()
            self.source_data_fetch_time = cycle_record
            attrcycle = onnx.helper.make_attribute("cycle_record", self.source_data_fetch_time)
            node.attribute.insert(-1, attrcycle)

            max_cycle = max(self.source_data_fetch_time)-2

            pooling_inst = PoolTmpl.format(
                                op_type = node.op_type,
                                ifmsize = thisifmsize,
                                Kernel = K,
                                Stride = S,
                                Padding = P,
                                source = ts,
                                destination = td
                            )
            inst_record_update(self.pooling_cycle,max_cycle,node_id,pooling_inst)

            mov_inst = MovTmpl.format(source=td,
                            destination=ts,
                            length=reduce(mul, get_attribute(node, 'ifmsize')[:]))
            inst_record_update(self.mov_cycle,max_cycle+1,node_id,mov_inst)

    def process_activFunc_node_xbm(self, node, node_id, OpSrc):
        thisifmsize = get_attribute(node, 'ifmsize')
        ofmsize = get_attribute(node, 'ofmsize')

        memoryaddr = get_attribute(node, 'memoryaddr')
        if not memoryaddr or len(memoryaddr) < 2:
            print(
                f"Exception: 'memoryaddr' attribute is missing or incomplete in node {node.name}. Skipping this node.")
            return
        ts = memoryaddr[0] # BufSrc
        td = memoryaddr[1] # BufDst

        self.source_data_fetch_time = get_attribute(self.onnx_model.graph.node[OpSrc[0]], 'cycle_record')

        cycle_record = torch.tensor(self.source_data_fetch_time) + 3.0 # Activation result ready time is ifm fetch_time + 3 cycles
        cycle_record = cycle_record.int().tolist()

        self.source_data_fetch_time = cycle_record
        attrcycle = onnx.helper.make_attribute("cycle_record", self.source_data_fetch_time)
        node.attribute.insert(-1, attrcycle)

        time = set(self.source_data_fetch_time) # # Unique ready times

        # Generate activation instructions for each unique fetch time
        for t in time:
            begin = self.source_data_fetch_time.index(t) # First occurrence of t
            end = len(self.source_data_fetch_time) - 1 - self.source_data_fetch_time[::-1].index(t) # Last occurrence of t
            activ_inst = ActivFuncTmpl.format(op_type = node.op_type,
                                        length = end - begin + 1, # length = Consecutive occurrences of 't' in the list
                                        source = ts + begin,
                                        destination = td + begin
                                    )

            inst_record_update(self.activ_cycle,t,node_id,activ_inst)

    def process_softmax_node_xbm(self, node, node_id, OpSrc):
        memoryaddr = get_attribute(node, 'memoryaddr')
        if not memoryaddr or len(memoryaddr) < 2:
            print(
                f"Exception: 'memoryaddr' attribute is missing or incomplete in node {node.name}. Skipping this node.")
            return
        ts = memoryaddr[0]
        td = memoryaddr[1]

        self.source_data_fetch_time = get_attribute(self.onnx_model.graph.node[OpSrc[0]], 'cycle_record')

        # Softmax needs max fetch_time from all ifm elements for global normalization
        max_fetch_time = max(self.source_data_fetch_time)
        cycle_record = len(self.source_data_fetch_time) * [max_fetch_time+ 3.0]

        self.source_data_fetch_time = cycle_record
        attrcycle = onnx.helper.make_attribute("cycle_record", self.source_data_fetch_time)
        node.attribute.insert(-1, attrcycle)

        softmax_inst = SoftmaxTmpl.format(length = td - ts + 1,source = ts,destination = td)

        inst_record_update(self.softmax_cycle,max_fetch_time,node_id,softmax_inst)


    def process_aluOp_node_xbm(self, node, node_id, OpSrc):
        # Initialize inp1 and inp2 to None or some default value
        inp1 = None
        inp2 = None

        ofmsize = get_attribute(node, 'ofmsize')
        if not OpSrc:
            print(f"Warning: No input sources (OpSrc) for ALU operation node {node.name}. Skipping this node.")
            return
        if len(OpSrc) > 1:
            tmp_record1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "cycle_record")
            tmp_record2 = get_attribute(self.onnx_model.graph.node[OpSrc[1]], "cycle_record")
        else:
            tmp_record1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "cycle_record")
            tmp_record2 = [0]
        td = get_attribute(node, 'memoryaddr')[-1] # BufDst

        # get the source data fetch time 
        if len(tmp_record1) == len(tmp_record2):
            cycle_record = np.maximum(np.array(tmp_record1, dtype=int) + 3,
                                      np.array(tmp_record2, dtype=int) + 3).tolist()
        else:
            if len(tmp_record1) < len(tmp_record2):
                if len(tmp_record1) == 0:
                    tmp_record1 = np.zeros_like(tmp_record2)
                else:
                    tmp_record1 = np.pad(np.array(tmp_record1), (0, len(tmp_record2) - len(tmp_record1)),
                                         'constant', constant_values=max(tmp_record1))
                tmp_record2 = np.array(tmp_record2)
            else:
                if len(tmp_record2) == 0:
                    tmp_record2 = np.zeros_like(tmp_record1)
                else:
                    tmp_record2 = np.pad(np.array(tmp_record2), (0, len(tmp_record1) - len(tmp_record2)),
                                         'constant', constant_values=max(tmp_record2))
                tmp_record1 = np.array(tmp_record1)
            cycle_record = (np.maximum(tmp_record1, tmp_record2) + 3).tolist()

        # record the output gain time
        if len(cycle_record) > 0:
           self.source_data_fetch_time = list(np.array(cycle_record))
        elif ofmsize:
           self.source_data_fetch_time = list(np.array([0] * reduce(mul, ofmsize[:])))
        attrcycle = onnx.helper.make_attribute("cycle_record",self.source_data_fetch_time)
        node.attribute.insert(-1, attrcycle)

        bufsrc = get_attribute(node, 'memoryaddr')[:-1]

        # inst generation
        if len(bufsrc) == 2:
            bufsrc1, bufsrc2 = bufsrc
            inp1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")
            if len(OpSrc) > 1:
                inp2 = get_attribute(self.onnx_model.graph.node[OpSrc[1]], "ofmsize")
                alu_inst = ALUopTmpl.format(op_type = node.op_type,
                                     input_source1 = bufsrc1,
                                     input_source2 = bufsrc2,
                                     input_length1 = reduce(mul, inp1[:]),
                                     input_length2 = reduce(mul, inp2[:]),
                                     destination = td)
            else:
                inp2 = self.initializer[get_attribute(node, 'srcinit').decode("utf-8")]['shape']
                alu_inst = ALUopTmpl.format(op_type = node.op_type,
                                     input_source1 = bufsrc1,
                                     input_source2 = bufsrc2,
                                     input_length1 = reduce(mul, inp1[:]),
                                     input_length2 = reduce(mul, inp2),
                                     destination = td)

        # Changed to assert for stricter validation; modified by [kevinJhz] on [2024-01-26].
        # Ensure that self.source_data_fetch_time is not empty; raise an assertion error if it is.
        assert len(self.source_data_fetch_time) > 0, "self.source_data_fetch_time is empty."

        max_cycle = max(self.source_data_fetch_time)
        if inp1 and inp2:
            inst_record_update(self.alu_cycle, max_cycle, node_id,alu_inst)

    def process_constant_node_xbm(self, node, node_id, OpSrc):
        OfmSize = get_attribute(node, 'ofmsize')
        if node.op_type == "Constant":# Constant
            cycle_record = reduce(mul, OfmSize[:]) * [0] # constant value is ready from cycle 0 to the end
        else:# Shape, ConstantOfShape, etc.
            if node.input[0] == 'input':
                cycle_record = reduce(mul, OfmSize[:]) * [0]  # 'input' value is ready from cycle 0 to the end
            else:
                cycle_record = torch.tensor(
                    self.source_data_fetch_time) + 0  # consider no computation in Shape,ConstantOfShape nodes
                cycle_record = cycle_record.int().tolist()

        self.source_data_fetch_time = cycle_record
        attrcycle = onnx.helper.make_attribute("cycle_record", self.source_data_fetch_time)
        node.attribute.insert(-1, attrcycle)

    def process_where_node_xbm(self,node,node_id,OpSrc):
        inp1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")
        inp2 = get_attribute(self.onnx_model.graph.node[OpSrc[1]], "ofmsize")
        inp3 = get_attribute(self.onnx_model.graph.node[OpSrc[2]], "ofmsize")
        bufsrc = get_attribute(node, 'memoryaddr')[:-1]
        bufdst = get_attribute(node, 'memoryaddr')[-1]

        ofmsize = get_attribute(node, 'ofmsize')

        inst = WhereTmpl.format(
            input_source1=str(bufsrc[0]),
            input_source2=str(bufsrc[1]),
            input_source3=str(bufsrc[2]),
            input_length1=str(reduce(mul, inp1[:])),
            input_length2=str(reduce(mul, inp2[:])),
            input_length3=str(reduce(mul, inp3[:])),
            ofmsize=str(ofmsize),
            destination=str(bufdst)
        )

        source_data_fetch_time_1 = get_attribute(self.onnx_model.graph.node[OpSrc[0]], 'cycle_record')
        source_data_fetch_time_2 = get_attribute(self.onnx_model.graph.node[OpSrc[1]], 'cycle_record')
        source_data_fetch_time_3 = get_attribute(self.onnx_model.graph.node[OpSrc[2]], 'cycle_record')

        # For transform-related operations, a maximum fetch_time from all ifm elements is required
        max_fetch_time = max(source_data_fetch_time_1 + source_data_fetch_time_2 + source_data_fetch_time_3)
        cycle_record = reduce(mul, ofmsize[:]) * [max_fetch_time+ 3.0]

        self.source_data_fetch_time = cycle_record
        attrcycle = onnx.helper.make_attribute("cycle_record", self.source_data_fetch_time)
        node.attribute.insert(-1, attrcycle)

        inst_record_update(self.where_cycle,max_fetch_time,node_id,inst)


    def process_elementwise_node_xbm(self, node, node_id, OpSrc):
        memoryaddr = get_attribute(node, 'memoryaddr')
        if not memoryaddr or len(memoryaddr) < 2:
            print(
                f"Exception: 'memoryaddr' attribute is missing or incomplete in node {node.name}. Skipping this node.")
            return
        ts = memoryaddr[0]  # BufSrc
        td = memoryaddr[1]  # BufDst

        self.source_data_fetch_time = get_attribute(self.onnx_model.graph.node[OpSrc[0]], 'cycle_record')

        cycle_record = torch.tensor(self.source_data_fetch_time) + 3.0  # Element-wise result ready time is ifm fetch_time + 3 cycles
        cycle_record = cycle_record.int().tolist()

        self.source_data_fetch_time = cycle_record
        attrcycle = onnx.helper.make_attribute("cycle_record", self.source_data_fetch_time)
        node.attribute.insert(-1, attrcycle)

        time = set(self.source_data_fetch_time)  # # Unique ready times

        # Generate activation instructions for each unique fetch time
        for t in time:
            begin = self.source_data_fetch_time.index(t)  # First occurrence of t
            end = len(self.source_data_fetch_time) - 1 - self.source_data_fetch_time[::-1].index(t)  # Last occurrence of t

            inst = ElementWiseTmpl.format(
                op_type=node.op_type,
                input_length=end - begin + 1,  # length = Consecutive occurrences of 't' in the list
                input_source=ts + begin,
                destination=td + begin
            )

            inst_record_update(self.activ_cycle, t, node_id, inst)

    def process_indexing_node_xbm(self, node, node_id, OpSrc):
        inst = None
        memoryaddr = get_attribute(node, 'memoryaddr')
        if not memoryaddr or len(memoryaddr) < 2:
            print(
                f"Exception: 'memoryaddr' attribute is missing or incomplete in node {node.name}. Skipping this node.")
            return

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

                inst = SliceTmpl.format(
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
        else:  # Gather
            if len(OpSrc) == 2:
                inp = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")
                indices = get_attribute(node, "indices")
                axis = get_attribute(node, "axis")
                ofm = get_attribute(node, "ofmsize")
                if inp is None:
                    inp = get_attribute(node, "inp_shape")
                inst = GatherTmpl.format(
                    input_source=str(bufsrc[0]),
                    input_length=str(reduce(mul, inp[:])),
                    ifmsize=str(inp),
                    indices = str(indices),
                    axis = str(axis),
                    ofmsize=str(ofm),
                    destination=str(bufdst)
                )
        self.source_data_fetch_time = get_attribute(self.onnx_model.graph.node[OpSrc[0]], 'cycle_record')

        # For indexing operations, assume that the maximum fetch_time is obtained from all ifm elements
        max_fetch_time = max(self.source_data_fetch_time)
        cycle_record = len(self.source_data_fetch_time) * [max_fetch_time + 3.0]

        self.source_data_fetch_time = cycle_record
        attrcycle = onnx.helper.make_attribute("cycle_record", self.source_data_fetch_time)
        node.attribute.insert(-1, attrcycle)

        inst_record_update(self.indexing_cycle, max_fetch_time, node_id, inst)

    def process_reduction_node_xbm(self, node, node_id, OpSrc):
        inp = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")  # data
        axes = get_attribute(node, "axes")
        ofmsize = get_attribute(node, 'ofmsize')

        memoryaddr = get_attribute(node, 'memoryaddr')
        if not memoryaddr or len(memoryaddr) < 2:
            print(
                f"Exception: 'memoryaddr' attribute is missing or incomplete in node {node.name}. Skipping this node.")
            return
        ts = memoryaddr[0]
        td = memoryaddr[1]

        inst = ReduceMeanTmpl.format(
            input_source=str(ts),
            input_length=str(reduce(mul, inp[:])),
            ifmsize=str(inp),
            axes=str(axes),
            ofmsize=str(ofmsize),
            destination=str(td)
        )

        self.source_data_fetch_time = get_attribute(self.onnx_model.graph.node[OpSrc[0]], 'cycle_record')
        max_fetch_time = max(self.source_data_fetch_time)
        cycle_record = len(self.source_data_fetch_time) * [max_fetch_time+ 3.0]

        self.source_data_fetch_time = cycle_record
        attrcycle = onnx.helper.make_attribute("cycle_record", self.source_data_fetch_time)
        node.attribute.insert(-1, attrcycle)

        inst_record_update(self.reduction_cycle,max_fetch_time,node_id,inst)

    def process_transform_node_xbm(self, node, node_id, OpSrc):
        inst = None
        memoryaddr = get_attribute(node, 'memoryaddr')
        if not memoryaddr or len(memoryaddr) < 2:
            print(
                f"Exception: 'memoryaddr' attribute is missing or incomplete in node {node.name}. Skipping this node.")
            return
        ts = memoryaddr[:-1]
        td = memoryaddr[-1]
        if len(ts) == 1:  # transpose or flatten
            if len(OpSrc) == 1:
                inp = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")
                ofm = get_attribute(node, "ofmsize")
                if node.op_type == "Transpose":
                    perm = get_attribute(node, "perm")
                    inst = TransposeTmpl.format(
                        input_source=str(ts[0]),
                        input_length=str(reduce(mul, inp[:])),
                        ifmsize=str(inp),
                        perm=str(perm),
                        ofmsize=str(ofm),
                        destination=str(td)
                        )
                elif node.op_type == "Flatten":
                    axis = get_attribute(node, "axis")
                    inst = FlattenTmpl.format(
                        input_source=str(ts[0]),
                        input_length=str(reduce(mul, inp[:])),
                        ifmsize=str(inp),
                        axis=str(axis),
                        ofmsize=str(ofm),
                        destination=str(td)
                        )
        elif len(ts) == 2: # reshape, expand, or unsqueeze
            OpSrc = GetInputOP(self.onnx_model, node_id)
            if len(OpSrc) == 2:
                inp = get_attribute(self.onnx_model.graph.node[OpSrc[0]], "ofmsize")
                ofm = get_attribute(node, "ofmsize")
                if node.op_type =='Reshape': # Reshape
                    inst = ReshapeTmpl.format(
                        op_type = str(node.op_type),
                        input_source = str(ts[0]),
                        input_length = str(reduce(mul,inp[:])),
                        ifmsize = str(inp),
                        ofmsize = str(ofm),
                        destination=str(td)
                    )
                elif node.op_type =='Expand':# Expad
                    inst = ExpandTmpl.format(
                        op_type = str(node.op_type),
                        input_source = str(ts[0]),
                        input_length = str(reduce(mul,inp[:])),
                        ifmsize = str(inp),
                        ofmsize = str(ofm),
                        destination=str(td)
                    )
                elif node.op_type =='Unsqueeze':# Unsqueeze
                    axes = get_attribute(node, "axes")
                    inst = UnsqueezeTmpl.format(
                        op_type = str(node.op_type),
                        input_source = str(ts[0]),
                        input_length = str(reduce(mul,inp[:])),
                        ifmsize = str(inp),
                        axes= axes,
                        ofmsize = str(ofm),
                        destination=str(td)
                    )

        self.source_data_fetch_time = get_attribute(self.onnx_model.graph.node[OpSrc[0]], 'cycle_record')

        # For transform-related operations, a maximum fetch_time from all ifm elements is required
        max_fetch_time = max(self.source_data_fetch_time)
        cycle_record = len(self.source_data_fetch_time) * [max_fetch_time+ 3.0]

        self.source_data_fetch_time = cycle_record
        attrcycle = onnx.helper.make_attribute("cycle_record", self.source_data_fetch_time)
        node.attribute.insert(-1, attrcycle)

        inst_record_update(self.transform_cycle,max_fetch_time,node_id,inst)

    def generate_output_instructions(self):
        keys = []
        keys.extend(self.compute_cycle.keys())
        keys.extend(self.pooling_cycle.keys())
        keys.extend(self.alu_cycle.keys())
        keys.extend(self.activ_cycle.keys())
        keys.extend(self.mov_cycle.keys())
        keys.extend(self.softmax_cycle.keys())
        keys_set = set(keys)
        for i in keys_set:
            if keys.count(i) > 1:
                print('flow{')
            if i in self.compute_cycle:
                for j in self.compute_cycle[i]:
                    if len(self.compute_cycle[i][j]) > 1:
                        print('\tparallel{')
                        for k in self.compute_cycle[i][j]:
                            print('\t\t', str(k))
                        print('\t}')
                    else:
                        for k in self.compute_cycle[i][j]:
                            print('\t\t', k)
            if i in self.pooling_cycle:
                for j in self.pooling_cycle[i]:
                    if len(self.pooling_cycle[i][j]) > 1:
                        print('\tparallel{')
                        for k in self.pooling_cycle[i][j]:
                            print('\t\t', k)
                        print('\t}')
                    else:
                        for k in self.pooling_cycle[i][j]:
                            print('\t\t', k)
            if i in self.alu_cycle:
                for j in self.alu_cycle[i]:
                    if len(self.alu_cycle[i][j]) > 1:
                        print('\tparallel{')
                        for k in self.alu_cycle[i][j]:
                            print('\t\t', k)
                        print('\t}')
                    else:
                        for k in self.alu_cycle[i][j]:
                            print('\t\t', k)
            if i in self.activ_cycle:
                for j in self.activ_cycle[i]:
                    if len(self.activ_cycle[i][j]) > 1:
                        print('\tparallel{')
                        for k in self.activ_cycle[i][j]:
                            print('\t\t', k)
                        print('\t}')
                    else:
                        for k in self.activ_cycle[i][j]:
                            print('\t\t', k)
            if i in self.mov_cycle:
                for j in self.mov_cycle[i]:
                    if len(self.mov_cycle[i][j]) > 1:
                        print('\tparallel{')
                        for k in self.mov_cycle[i][j]:
                            print('\t\t', k)
                        print('\t}')
                    else:
                        for k in self.mov_cycle[i][j]:
                            print('\t\t', k)
            if i in self.softmax_cycle:
                for j in self.softmax_cycle[i]:
                    if len(self.softmax_cycle[i][j]) > 1:
                        print('\tparallel{')
                        for k in self.softmax_cycle[i][j]:
                            print('\t\t', k)
                        print('\t}')
                    else:
                        for k in self.softmax_cycle[i][j]:
                            print('\t\t', k)
            if i in self.elementwise_op_cycle:
                for j in self.elementwise_op_cycle[i]:
                    if len(self.elementwise_op_cycle[i][j]) > 1:
                        print('\tparallel{')
                        for k in self.elementwise_op_cycle[i][j]:
                            print('\t\t', k)
                        print('\t}')
                    else:
                        for k in self.elementwise_op_cycle[i][j]:
                            print('\t\t', k)
            if i in self.indexing_cycle:
                for j in self.indexing_cycle[i]:
                    if len(self.indexing_cycle[i][j]) > 1:
                        print('\tparallel{')
                        for k in self.indexing_cycle[i][j]:
                            print('\t\t', k)
                        print('\t}')
                    else:
                        for k in self.indexing_cycle[i][j]:
                            print('\t\t', k)
            if i in self.reduction_cycle:
                for j in self.reduction_cycle[i]:
                    if len(self.reduction_cycle[i][j]) > 1:
                        print('\tparallel{')
                        for k in self.reduction_cycle[i][j]:
                            print('\t\t', k)
                        print('\t}')
                    else:
                        for k in self.reduction_cycle[i][j]:
                            print('\t\t', k)
            if i in self.transform_cycle:
                for j in self.transform_cycle[i]:
                    if len(self.transform_cycle[i][j]) > 1:
                        print('\tparallel{')
                        for k in self.transform_cycle[i][j]:
                            print('\t\t', k)
                        print('\t}')
                    else:
                        for k in self.transform_cycle[i][j]:
                            print('\t\t', k)
            if i in self.where_cycle:
                for j in self.where_cycle[i]:
                    if len(self.where_cycle[i][j]) > 1:
                        print('\tparallel{')
                        for k in self.where_cycle[i][j]:
                            print('\t\t', k)
                        print('\t}')
                    else:
                        for k in self.where_cycle[i][j]:
                            print('\t\t', k)
            if keys.count(i) > 1:
                print('}')