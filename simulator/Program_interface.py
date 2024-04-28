import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
sys.path.append('../configs')
from configs import *

class CIMInst():
    def __init__(self,chip=None) -> None:
        self.chip = chip
 
    def Conv(self,ifmsize,kernel,stride,padding,CoreAddr,ifmReadStart,OfmWriteStart,chip):      
        inp_b = torch.tensor(chip.GBBuf[ifmReadStart:ifmReadStart+math.prod(ifmsize)].reshape((ifmsize[0],ifmsize[2],ifmsize[3],ifmsize[1])).transpose(0,3,1,2))
        weight = torch.tensor(chip.core[CoreAddr].xbdata[:math.prod(kernel)]).reshape(kernel)
        inp = F.pad(inp_b,padding)
        data = F.conv2d(inp,weight,stride=stride)
        ofmsize = int(math.prod(data.size()))
        chip.GBBuf[OfmWriteStart:OfmWriteStart+ofmsize]=np.array(data).transpose(0,2,3,1).reshape(-1)
        return inp_b

    def FC(self,ifmlen,hidlen,ofmlen,CoreAddr,BsrcAddr,BdstAddr,WAddr1,WAddr2,chip):
        inp = torch.tensor(chip.GBBuf[BsrcAddr:BsrcAddr+ifmlen])
        weight1 = torch.tensor(chip.core[CoreAddr].xbdata[WAddr1:WAddr1+ifmlen*hidlen]).reshape((hidlen,ifmlen))
        tmp= F.linear(inp,weight1)
        weight2 = torch.tensor(chip.core[CoreAddr].xbdata[WAddr2:WAddr2+ofmlen*hidlen]).reshape((ofmlen,hidlen)) 
        data = F.linear(tmp,weight2)
        chip.GBBuf[BdstAddr:BdstAddr+ofmlen] = np.array(data)
        return data

    def DepConv(self,ifmsize,kernel,stride,padding,CoreAddr,ifmReadStart,OfmWriteStart,chip):      
        inp_b = torch.tensor(chip.GBBuf[ifmReadStart:ifmReadStart+math.prod(ifmsize)].reshape((ifmsize[0],ifmsize[2],ifmsize[3],ifmsize[1])).transpose(0,3,1,2))
        weight = torch.tensor(chip.core[CoreAddr].xbdata[:math.prod(kernel)]).reshape(kernel)
        inp = F.pad(inp_b,padding)
        data = F.conv2d(inp,weight,stride=stride,groups=ifmsize[1])
        ofmsize = int(math.prod(data.size()))
        chip.GBBuf[OfmWriteStart:OfmWriteStart+ofmsize]=np.array(data).transpose(0,2,3,1).reshape(-1)
        return inp_b

   
    def MV(self,chip,mAddr,vAddr,dAddr,vsize=None):        
        matrix = chip.core[mAddr[0]].xb[mAddr[1]].xb_data
        vec = chip.core[mAddr[0]].xb[mAddr[1]].reg[vAddr:vAddr+matrix.shape[0]]
        data = np.array(vec).dot(np.array(matrix))
        chip.core[mAddr[0]].xb[mAddr[1]].reg[dAddr:dAddr+len(data)] = data
        return data 
    
    def read_row(self,src,dst,srcAddr,dstAddr,length):
        assert len(src[srcAddr:])>= length and len(dst[dstAddr:])>=length, \
            (len(src[srcAddr:]),len(dst[dstAddr:]),length)
        dst[dstAddr:dstAddr+length] = src[srcAddr:srcAddr+length]
        return src[srcAddr:srcAddr+length]

class DCOMInst():
    def shift(self,dirct,src,dst,srcAddr,dstAddr,length,bit):
        assert len(src[srcAddr:])>= length and len(dst[dstAddr:])>=length, \
            (len(src[srcAddr:]),len(dst[dstAddr:]),length)
        if dirct == 'right':
            dst[dstAddr:dstAddr+length] = src[srcAddr:srcAddr+length]>>bit
        elif dirct == 'left':
            dst[dstAddr:dstAddr+length] = src[srcAddr:srcAddr+length]<<bit
        else:
            print("error: direction is %s, shift direction must in right/left"%(dirct))

    def add(self,src1,src2,dst,src1Addr,src2Addr,dstAddr,length):
        assert len(src1[src1Addr:])>= length and len(src2[src2Addr:])>= length and len(dst[dstAddr:])>=length, \
            (len(src1[src1Addr:]),len(src2[src2Addr:]),len(dst[dstAddr:]),length)
        dst[dstAddr:dstAddr+length] = src1[src1Addr:src1Addr+length]+\
                                        src2[src2Addr:src2Addr+length]
        return 0

    def sub(self,src1,src2,dst,src1Addr,src2Addr,dstAddr,length):
        assert len(src1[src1Addr:])>= length and len(src2[src2Addr:])>= length and len(dst[dstAddr:])>=length, \
            (len(src1[src1Addr:]),len(src2[src2Addr:]),len(dst[dstAddr:]),length)
        dst[dstAddr:dstAddr+length] = src1[src1Addr:src1Addr+length]-\
                                        src2[src2Addr:src2Addr+length]
        return 0
    
    def pooling(op,ifmsize,kernel,stride,padding,src,dst,ifmReadStart,OfmWriteStart):
        inp = torch.tensor(src[ifmReadStart:ifmReadStart+math.prod(ifmsize)].reshape((ifmsize[0],ifmsize[2],ifmsize[3],ifmsize[1])).transpose(0,3,1,2))
        outp = F.max_pool2d(inp,kernel,stride,padding)
        ofmsize = int(math.prod(outp.size()))
        dst[OfmWriteStart:OfmWriteStart+ofmsize]=np.array(outp).transpose(0,2,3,1).reshape(-1)
        return 0


    def func(self,op,src,dst,srcAddr,dstAddr,length):
        assert len(src[srcAddr:])>= length and len(dst[dstAddr:])>=length, \
            (len(src[srcAddr:]),len(dst[dstAddr:]),length)
        if op=='Relu':
            dst[dstAddr:dstAddr+length] = np.array(F.relu(torch.tensor(src[srcAddr:srcAddr+length])))
        elif op == 'Sigmoid':
            dst[dstAddr:dstAddr+length] = np.array(F.sigmoid(torch.tensor(src[srcAddr:srcAddr+length])))
        elif op == 'Tanh':
            dst[dstAddr:dstAddr+length] = np.array(F.tanh(torch.tensor(src[srcAddr:srcAddr+length])))
        return 0
    
class DMOVInst():
    def copy(self,src,dst,srcAddr,dstAddr,length,stride=1):
        dst[dstAddr:dstAddr+length:stride] = src[srcAddr:srcAddr+length:stride]
        return 0

    def load(self,op,src,dst,srcAddr,dstAddr,length,stride=1):
        if op == 'LCBuf':
            return src.LCBuf[srcAddr:srcAddr+length:stride]
        elif op == 'GBBuf':
            for sdata,ddata in zip(src,dst):
                assert len(sdata.GBBuf[srcAddr:])>=length and len(ddata.reg[dstAddr:])>=length, \
                    (len(sdata.GBBuf[srcAddr:]),len(ddata.reg[dstAddr:]),length)
                ddata.reg[dstAddr:dstAddr+length:stride] = sdata.GBBuf[srcAddr:srcAddr+length:stride]
        else:
            print('can not load data from %s to reg'%(str(op)))

    def store(self,op,src,dst,srcAddr,dstAddr,length,stride):
        if op == 'LCBuf':
            for sdata,ddata in zip(src,dst):
                assert len(sdata.reg[srcAddr:])>=length and len(ddata.LCBuf[dstAddr:])>=length, \
                    (len(sdata.reg[srcAddr:]),len(ddata.LCBuf[dstAddr:]),length)
                ddata.LCBuf[dstAddr:dstAddr+length:stride] = sdata.reg[srcAddr:srcAddr+length:stride]
        elif op == 'GBBuf':
            for sdata,ddata in zip(src,dst):
                assert len(sdata.reg[srcAddr:])>=length and len(ddata.GBBuf[dstAddr:])>=length, \
                    (len(sdata.reg[srcAddr:]),len(ddata.GBBuf[dstAddr:]),length)
                ddata.GBBuf[dstAddr:dstAddr+length:stride] = sdata.reg[srcAddr:srcAddr+length:stride]
        else:
            print('can not store data from reg to %s'%(str(op)))
    
    
# test
if __name__ == '__main__':
    chip = CreateCIM(ArchTem)
    CIM = CIMInst(chip)
    DCOM = DCOMInst()
    DMOV = DMOVInst()

    # get the input & weight data

    # parse instructions

    # simulate the CIM computation process by using the provided operators
