import math
import onnx
from utils.util import get_attribute

class CoreVirtualMapping:
    def __init__(self, CIM):
        # Initialize core and xb sizes from CIM
        self.core_size = len(CIM.core[0].xb)  # Number of xbs per core
        self.xb_size = CIM.core[0].xb[0].Size  # Size of a single xb [num_xbr, num_xbc]
        self.xb_bus = CIM.core[-1].LCBufBW  # Buffer bandwidth per core

    def virtual_mapping_conv(self, node):
        k =  get_attribute(node,'kernel_shape')[0]
        ifm = get_attribute(node,'ifmsize')
        ofm = get_attribute(node,'ofmsize')

        # Calculate the number of required xb rows based on the kernel size and IFM depth( xbr = (k * k * C_in) / num_xbr )
        xbr = math.ceil(k * k * ifm[1] / self.xb_size[0])
        # Calculate the number of required xb columns based on the OFM depth( xbc = C_out / num_xbc )
        xbc = math.ceil(ofm[1] / self.xb_size[1])
        # Calculate the number of required cores based on the xb rows and columns(core_num = (xbr * xbc) / core_size)
        r = math.ceil(xbr * xbc / self.core_size)
        # Calculate the latency based on the OFM size and a fixed coefficient(c = W_out * H_out * 8)
        c = ofm[2] * ofm[3] * 8
        # Calculate the maximum duplication based on the core's buffer bandwidth and kernel size( max_dup = xb_bus / ( k * k * C_in ) )
        max_dup = max(math.floor(self.xb_bus / (k * k * ifm[1])), 1)

        attrcore = onnx.helper.make_attribute("core_num",r)
        attrxb = onnx.helper.make_attribute("xb_num",[xbr,xbc])
        attrc = onnx.helper.make_attribute("latency",c)
        attrdup = onnx.helper.make_attribute('max_dup',max_dup)
        node.attribute.insert(-1,attrxb)
        node.attribute.insert(-1,attrcore)
        node.attribute.insert(-1,attrc)
        node.attribute.insert(-1,attrdup)

    def virtual_mapping_pool(self,node):
        k =  get_attribute(node,'kernel_shape')
        ifm = get_attribute(node,'ifmsize')
        ofm = get_attribute(node,'ofmsize')
        # latency = C_out * H_out * W_out * 4
        c = ofm[1]*ofm[2]*ofm[3]*4
        # core_num = 0
        attrr = onnx.helper.make_attribute("core_num",0)
        attrc = onnx.helper.make_attribute("latency",c)
        node.attribute.insert(-1,attrr)
        node.attribute.insert(-1,attrc)

    def virtual_mapping_matmul(self,node):
        ifm = get_attribute(node,'ifmsize')
        ofm = get_attribute(node,'ofmsize')
        # c = M = H_in、N = W_in、K = W_out
        M,N,K = ifm[-2],ifm[-1],ofm[-1]
        c = M
        # Adjust crossbar dimensions for non-standard MatMul inputs(like 2D matrixs).
        if node.op_type == 'MatMul' and len(ifm) != 3:
            # Calculate xb rows and columns based on batch size, input channels, and crossbar size.
            xbr = math.ceil(N * ifm[1] / self.xb_size[0])
            xbc = math.ceil(K * ofm[1] / self.xb_size[1])
            # Calculate required cores based on adjusted xb rows and columns.
            r = math.ceil(xbr * xbc / self.core_size)
        else:
            # Calculate xb rows and columns without considering batch size or channels.
            xbr = math.ceil(N / self.xb_size[0])
            xbc = math.ceil(K / self.xb_size[1])
            r = math.ceil(xbr * xbc / self.core_size)

        attrcore = onnx.helper.make_attribute("core_num",r)
        attrxb = onnx.helper.make_attribute("xb_num",[xbr,xbc])
        attrc = onnx.helper.make_attribute("latency",c)
        node.attribute.insert(-1,attrxb)
        node.attribute.insert(-1,attrcore)
        node.attribute.insert(-1,attrc)

    def get_mapping_attributes(self,onnx_model,ifmsize):
        # Handle the special case for the first node
        first_node = onnx_model.graph.node[0]
        ifm_size_attr = get_attribute(first_node, 'ifmsize')
        if ifm_size_attr is None:
            # Add 'ifmsize' attribute to the first node
            ifmsize_attribute = onnx.helper.make_attribute("ifmsize", list(ifmsize))
            first_node.attribute.append(ifmsize_attribute)

        for node_id, node in enumerate(onnx_model.graph.node):
            if node.op_type == 'Conv':
                # Compute Conv node attributes: core_num, xb_num, latency, max_dup
                self.virtual_mapping_conv(node)
            elif node.op_type in ['MaxPool', 'AveragePool']:
                # Compute Pooling node attributes: core_num, latency
                self.virtual_mapping_pool(node)
            elif node.op_type in ['MatMul', 'Gemm']:
                # Compute MatMul/Gemm node attributes: core_num, xb_num, latency
                self.virtual_mapping_matmul(node)


