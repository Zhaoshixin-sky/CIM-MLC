import numpy as np 
from .example_config import *

class ChipLevel:
    def __init__(self):
        self.core = []
        self.GBBuf = []
        self.alu = 0
        self.noctype='SM'
        self.noccost=None
    def create_core(self,core):
        self.core.append(core)
    def create_noc(self,noctype='SM',noccost=None):
        self.noctype=noctype
        self.noccost = noccost  
    def create_buf(self,buf):
        self.GBBuf=buf
    def create_alu(self,alu):
        self.alu=alu
    def creste_bus(self,bus):
        self.GBBufBW = bus    

class CoreLevel:
    def __init__(self,Addr):
        self.Addr=Addr
        self.xb = []
        self.LCBuf = np.zeros((1,1))    #TODO
        self.alu = 0
        self.noctype='SM'
        self.noccost=None
        self.flag=0
    def create_xb(self,xb):
        self.xb.append(xb)
    def create_noc(self,noctype='SM',noccost=None):
        self.noctype=noctype
        self.noccost = noccost  
    def create_buf(self,buf):
        self.LCBuf=buf
    def create_alu(self,alu):
        self.alu=alu
    def create_bus(self,bus):
        self.LCBufBW = bus  

class XBLevel:
    def __init__(self,Addr,Size,MaxRC,device):
        self.Addr = Addr
        self.Size = Size
        self.MaxR = MaxRC
        self.device = device

class DeviceLevel:
    def __init__(self,Type,Precision):
        self.Type = Type
        self.Precision = Precision 

def CreateCIM(ArchTem):
    Chip = ChipLevel()
    for i in range(ArchTem["CoreNum"][0]*ArchTem["CoreNum"][1]):
        Chip.create_core(CoreLevel(i))
    Chip.create_noc(ArchTem['CoreNoc'],ArchTem['CoreNocCost'])
    Chip.create_buf(ArchTem['GBBuf'])
    Chip.create_alu(ArchTem['CoreALU'])
    Chip.creste_bus(ArchTem['CoreBus'])

    for core in Chip.core:
        for i in range(ArchTem["XBNum"][0]*ArchTem["XBNum"][1]):
            core.create_xb(XBLevel(
                i,
                ArchTem["XBSize"],
                ArchTem["MaxRC"],
                DeviceLevel(ArchTem['Type'],ArchTem['Precision'])))
        core.create_noc(ArchTem['XBNoc'],ArchTem['XBNocCost'])
        core.create_buf(ArchTem['LCBuf'])
        core.create_alu(ArchTem['XBALU'])
        core.create_bus(ArchTem['XBbus'])
    return Chip