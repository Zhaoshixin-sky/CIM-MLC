import math 
from utils.util import *

class CrossbarWiseDup:
    def __init__(self,CIM, DupSeg):
        self.CIM = CIM
        self.DupSeg = DupSeg

    def crossbarwise_updata_dup(self,ttype = 'real'):
        XBperCore = len(self.CIM.core[0].xb)
        for CurDupSeg in self.DupSeg:
            for node in CurDupSeg:
                if node.op_type== 'Conv':
                    XBNum = get_attribute(node,'xb_num')[0]*get_attribute(node,'xb_num')[1]
                    CoreNum = get_attribute(node,'core_num')
                    DupNum = get_attribute(node,'dup')
                    XBWiseDupNum = max(math.floor(CoreNum*DupNum*XBperCore/XBNum),1)
                    if ttype=='test':
                        attrxbdup = onnx.helper.make_attribute('xdup',1)
                    else:
                        attrxbdup = onnx.helper.make_attribute('xdup',XBWiseDupNum)
                    node.attribute.extend([attrxbdup])

