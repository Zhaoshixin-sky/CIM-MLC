from copy import deepcopy
import numpy as np 
class TreeNode:
    def __init__(self, tag, type, param):
        self.tag = tag
        self.type = type
        self.resource =0 
        self.latency =0 
        self.dup=0
        self.child = []
        self.param = param
    def create_child(self,node):
        self.child.append(node)
    def Getinfom(self):
        print([self.tag, self.type, self.resource, self.param,self.dup,self.child ])

def GetAllPaths(root):
    def construct_paths(root, path):
        if not root: return 
        path.append(root)
        if len(root.child)==0:     
            paths.append(path)
        else:
            for node in root.child:
                construct_paths(node,path.copy())
    paths = []
    construct_paths(root, [])
    return paths

def GetLongestNode(AllPath):
    max_latency = []
    for trace in AllPath:
        sum_latency = 0
        for n in trace:
            sum_latency+=n.latency
        max_latency.append(sum_latency)
    MaxLTrace = AllPath[max_latency.index(max(max_latency))]# 最长的路径
    TmpList = [node.latency - node.base_latency / ((node.resource + node.base_resource)/node.base_resource) for node in MaxLTrace]#原来的延迟-复制后延迟，提升越大，性价比越高
    MaxLNode = MaxLTrace[TmpList.index(max(TmpList))]
    return MaxLTrace, MaxLNode, max(max_latency)

def GetMaxActR(AllPath):
    MaxActR = 0
    for trace in AllPath:
        MaxActR+=max([n.resource for n in trace])
    return MaxActR-2
