# input : network model, hardware architecture
# output: operators duplication result in core mode
import onnx
import numpy as np
from utils.util import get_attribute

class CoreWiseDuper:
    def __init__(self,TotalCore):
        self.TotalCore = TotalCore
        self.FlowSeg = []    # network segment record
        self.CurFlowSeg = [] # current network segment
        self.DupSeg = []     # network segment record for nodes need computation resources
        self.CurDupSeg = []  # current network segment for nodes need computation resources
        self.NeededCore = 0

    def seg_conv(self, node):
        # segment conv node
        # Check if adding the current node would exceed the total number of cores.
        if self.NeededCore + get_attribute(node, 'core_num') <= self.TotalCore:
            # If not, append the node to both the current flow and duplication segments, which means the node can be processed in the current resource allocation.
            self.CurFlowSeg.append(node)
            self.CurDupSeg.append(node)
            # Update the total demand for the Core
            self.NeededCore += get_attribute(node, 'core_num')
        else:
            # Start a new computing segment if resources are exhausted, and save the current FlowSeg and DupSeg.
            self.FlowSeg.append(self.CurFlowSeg)
            self.DupSeg.append(self.CurDupSeg)
            # Start a new FlowSeg and DupSeg containing only the current node.
            self.CurDupSeg = [node]
            self.CurFlowSeg = [node]
            self.NeededCore = get_attribute(node, 'core_num')
    
    def seg_matmul(self,node):
        # Matmul nodes only updates the FlowSeg
        if self.NeededCore + get_attribute(node,'core_num') <= self.TotalCore:
                self.CurFlowSeg.append(node)
                self.NeededCore += get_attribute(node,'core_num')
        else:
            self.FlowSeg.append(self.CurFlowSeg)
            self.CurFlowSeg = [node]
            self.NeededCore = get_attribute(node,'core_num')
        attrdup = onnx.helper.make_attribute("dup", 1)            
        node.attribute.insert(-1,attrdup)
    
    def seg_pool(self,node):
        # add pooling to current FlowSegment in defult
        # Pooling nodes do not occupy core and do not need to be copied.
        self.CurFlowSeg.append(node)

    def create_network_segments(self,onnx_model):
        for node_id, node in enumerate(onnx_model.graph.node):
            if node.op_type == 'Conv':
                self.seg_conv(node)
            elif node.op_type in ['MaxPool', 'AveragePool']:
                self.seg_pool(node)
            elif node.op_type in ['MatMul', 'Gemm']:
                self.seg_matmul(node)
            else:
                continue
        # Record the last moment of FlowSeg and DupSeg after traversing all the nodes,
        self.FlowSeg.append(self.CurFlowSeg)
        self.DupSeg.append(self.CurDupSeg)
        return self.FlowSeg, self.DupSeg

    def get_corewise_dup(self, DupType = 'real'):
        # Allocate resources using dynamic programming
        for CurDupSeg,CurFlowSeg in zip(self.DupSeg,self.FlowSeg):
            CoreNum = [0]
            Latency = [0]
            ofmw = [0]
            NeedCoreNum = 0

            # Accumulate core numbers required for MatMul and Gemm operators
            for node in CurFlowSeg:
                if node.op_type == 'MatMul' or node.op_type == 'Gemm':
                    NeedCoreNum += get_attribute(node, 'core_num')

            # Collect core_num, latency, and ofmw for each node in the dup segment
            for node in CurDupSeg:
                CoreNum.append(get_attribute(node, 'core_num'))
                Latency.append(get_attribute(node, 'latency'))
                ofmw.append(get_attribute(node, 'ofmsize')[2])

            # Calculate total remaining computation resources
            # Remaining core resources = Global core resources - (core_num occupation of the MatMul or Gemm node) - (core_num occupation of the dupseg node)
            TotalResc = self.TotalCore - NeedCoreNum - sum(CoreNum)  # Remaining computation resources
            # layerN = total number of nodes in the CurDupSeg
            LayerN = len(CoreNum) - 1

            if TotalResc >= 0:
                # Perform dynamic programming to allocate remaining cores
                LayerN = len(CoreNum) - 1
                path = np.ones((LayerN + 1, TotalResc + 1)) # Path matrix : path[i] denotes the number of cores assigned in the preceding layer when computing the optimal delay for layer i.
                D = np.zeros((LayerN + 1, TotalResc + 1)) # Duplication matrix
                dp = np.ones((LayerN + 1, TotalResc + 1)) * sum(Latency) # dp matrix : Records minimum delays for different layer and remaining resource (r) conditions.

                # DP process : determine optimal core allocation and duplication strategy
                for layer in range(1, LayerN + 1):
                    if CoreNum[layer] > 0:
                        for r in range(0, TotalResc + 1):  # Iterate over the remaining resources, r is the current remaining resource
                            if r >= CoreNum[layer]:
                                for k in range(1, int(r / CoreNum[layer]) + 1):  # Iterate over all possible duplication times k
                                    # Check if the current replication time is legal, make sure it can be divided by ofmw[layer].
                                    if ofmw[layer] % (k + 1) == 0:
                                        # Update the latency information in the dynamic programming matrix
                                        dp[layer][r] = min(dp[layer][r],
                                                           # dp[layer - 1][r - k * CoreNum[layer]]  : The minimum latency of the previous layer node layer - 1 under the remaining resources r - k * CoreNum[layer], k is the number of replications
                                                           # Latency[layer] / (k + 1)               : Latency of replicating node layer
                                                           # Latency[layer]                         : Latency of not replicating node layer
                                                           dp[layer - 1][r - k * CoreNum[layer]] + Latency[layer] / (k + 1) - Latency[layer])

                                        # Update path and duplication matrix if current duplication leads to minimum latency
                                        if dp[layer][r] == dp[layer - 1][r - k * CoreNum[layer]] + Latency[layer] / (k + 1) - Latency[layer]:
                                            D[layer][r] = k
                            else:
                                # If current resource is insufficient, use previous layer's allocation
                                dp[layer][r] = dp[layer - 1][r]
                            # Ensure dp[layer][r] is updated to the minimum value in each step
                            dp[layer][r] = min(dp[layer - 1][r], dp[layer][r])


                # Backtracking process : determine the optimal number of replications and path
                RescTmp = TotalResc
                Dup = np.zeros(LayerN + 1) # Initialize the duplication count matrix
                # Backtrack from the last layer to update duplication and path
                for layer in range(LayerN, 0, -1):
                    if CoreNum[layer] > 0:  # Skip replication and path update if current layer doesn't need core resources
                        # Start from the current remaining resources and move forward
                        for r in range(RescTmp, 0, -1):
                            k = int(D[layer][r]) # Get the optimal replication count from the previous DP step.
                            if r >= CoreNum[layer]:# If the current resources are enough for the layer's core, proceed.
                                # If current DP value is optimal, update path, duplication, and remaining resources.
                                if dp[layer][RescTmp] == dp[layer - 1][r - k * CoreNum[layer]] + Latency[layer] / (k + 1) - Latency[layer]:
                                    # Update path matrix for current layer, reflecting the optimal path with remaining resources r and replication k.
                                    path[layer] = r - k * CoreNum[layer] # r - k * CoreNum[layer] represents the new remaining resources after allocation.
                                    Dup[layer] = k
                                    RescTmp -= k * CoreNum[layer]
                                    break


            # Apply DP results to update core usage and replication counts for each node.
            DefaultDup = [1] * LayerN
            for node_index in range(1, LayerN + 1):
                node = CurDupSeg[node_index - 1]
                attrusedcore = onnx.helper.make_attribute("usedcore", int(Dup[node_index]) + 1)
                node.attribute.insert(-1, attrusedcore)
                if DupType == 'test':
                    attrdup = onnx.helper.make_attribute("dup", DefaultDup[node_index - 1])
                else:
                    attrdup = onnx.helper.make_attribute("dup", int(Dup[node_index]) + 1)
                node.attribute.insert(-1, attrdup)

        for CurFlowSeg in self.FlowSeg:
            # core physical mapping
            CoreStartID = 0
            for node in CurFlowSeg:
                if node.op_type == 'Conv' or node.op_type == 'MatMul' or node.op_type == 'Gemm':
                    NodeCoreNum = get_attribute(node, 'core_num') * get_attribute(node, 'dup')
                    attrcorestart = onnx.helper.make_attribute("corestart", CoreStartID)
                    node.attribute.insert(-1, attrcorestart)
                    CoreStartID = CoreStartID + NodeCoreNum