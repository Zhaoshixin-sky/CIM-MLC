import math
import onnx
from utils.util import get_attribute

# pipline balance
class CoreWisePipeline:
    def __init__(self, onnx_model, CIM, FlowSeg):
        self.onnx_model = onnx_model
        self.CIM = CIM
        self.FlowSeg = FlowSeg

    def update_dup(self):
        # Iterate over each flow segment
        for CurFlowSeg in self.FlowSeg:
            # Get original duplication counts for all nodes in CurFlowSeg
            OriginalDup = self.get_original_duplicates(CurFlowSeg)

            # Initialize balanced duplicate counts
            BalanceDup = [get_attribute(CurFlowSeg[0], 'dup')]
            DupScale = 1  # Initialize balancing factor to 1

            # Iterate over each node in CurFlowSeg
            for node_index in range(1, len(CurFlowSeg)):
                # Get the node's stride if exists, otherwise default to 1
                stride = get_attribute(CurFlowSeg[node_index], 'strides')
                stride = stride[0] if stride else 1
                DupScale *= stride  # Update DupScale by multiplying with stride

                # If the current node is a convolution layer
                if CurFlowSeg[node_index].op_type == 'Conv':
                    # Calculate new duplication count
                    t_dup = min(math.floor(BalanceDup[0] / DupScale), get_attribute(CurFlowSeg[node_index], 'dup'))

                    # If new duplication count is greater than the last element of the list (previous node's duplication)
                    if t_dup > BalanceDup[-1]:
                        BalanceDup.append(max(math.floor(BalanceDup[-1] / stride), 1))
                    else:
                        BalanceDup.append(max(t_dup, 1))

                    # Update 'dup' attribute of the current node
                    found = [attr_id for attr_id, attr in enumerate(CurFlowSeg[node_index].attribute) if
                             attr.name == 'dup']
                    new_dup = onnx.helper.make_attribute("dup", BalanceDup[-1])
                    del CurFlowSeg[node_index].attribute[found[0]]
                    CurFlowSeg[node_index].attribute.extend([new_dup])

    def get_original_duplicates(self, cur_flow_seg):
        original_dup = []
        for node_index in range(0, len(cur_flow_seg)):
            if cur_flow_seg[node_index].op_type == 'Conv':
                original_dup.append(get_attribute(cur_flow_seg[node_index], 'dup'))
        return original_dup
