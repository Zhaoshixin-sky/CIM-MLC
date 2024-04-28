import numpy as np
import onnx
import sys
sys.path.append('../configs')
sys.path.append('../optimization')
sys.path.append('../onnx_model')
sys.path.append('../utils')
from configs import *
from optimization import *
from utils import *
from .PreformSim import *

def run(modelname,CIM):
    onnx_model = onnx.load(modelname)
    # get middle feature map information
    ort_outs = get_layer_output(onnx_model,np.random.randn(1, 3,32, 32).astype(np.float32))
    ifmsize =(1,3,32,32)

    FlowSeg,DupSeg = CoreVirtualMapping(onnx_model,ort_outs,CIM,ifmsize)
    baseline_l = get_latency(onnx_model,CIM,FlowSeg,test_type = 'baseline')[0]

    CoreWiseDup(onnx_model,CIM,DupSeg,ttype = 'real')
    cd_l = get_latency(onnx_model,CIM,FlowSeg,test_type='cd')[0]

    CoreWisepipeline(onnx_model,CIM,FlowSeg)
    cp_l = get_latency(onnx_model,CIM,FlowSeg,test_type='cp')[0]
    cdcp_l,_,_ = get_latency(onnx_model,CIM,FlowSeg,test_type='cdcp')

    crossbarWiseDup(onnx_model, CIM, DupSeg,ttype = 'real')        
    xdcp_l,_,_ = get_latency(onnx_model,CIM,FlowSeg,test_type='xdcp')
    xdxp_l,_,_ = get_latency(onnx_model,CIM,FlowSeg,test_type='xdxp')
    xdwlp_l,_,_ = get_latency(onnx_model,CIM,FlowSeg,test_type='xdwlp')


if __name__ == '__main__':
    CIM = CreateCIM(ArchTem)
    data = []
    name = '' # model name 
    run(name,CIM)

