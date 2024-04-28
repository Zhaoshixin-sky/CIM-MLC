import math
from utils.util import get_attribute

def get_latency(CIM,FlowSeg,test_type = 'baseline'):    
    ReadyTimeRecord,EndTimeRecord = [],[]
    alu = CIM.core[-1].alu
    GBufBW = CIM.GBBufBW
    xbr,xbc = CIM.core[0].xb[0].Size[0], CIM.core[0].xb[0].Size[1]
    BufBW = CIM.core[-1].LCBufBW
    MaxR = CIM.core[0].xb[0].MaxR
    ComptTimeScale = math.ceil(xbr/MaxR)
    XBperCore = len(CIM.core[0].xb)
    latency = []
    FlashTime = 0
    if test_type == 'baseline':    
        for CurFlowSeg in FlowSeg:
            pre_layer ='N' #begin with the first layer
            EndTime = 0
            for node in CurFlowSeg:
                ReadyTime = EndTime
                if node.op_type == 'Conv':
                    k =  get_attribute(node,'kernel_shape')[0]
                    cin = get_attribute(node,'ifmsize')[1]
                    cout = get_attribute(node,'ofmsize')[1]
                    ifm = get_attribute(node,'ifmsize')[2]
                    ofm = get_attribute(node,'ofmsize')[2]
                    
                    GBLoadTime = math.ceil(cin/GBufBW)
                    LBufBW = math.ceil(k*k*cin/BufBW )
                    ComptTime = get_attribute(node,'latency')
                    EndTime = ReadyTime + GBLoadTime + LBufBW + ComptTime*ComptTimeScale +  math.ceil(cout/BufBW) + math.ceil(cout/GBufBW)
                    '''
                    权重刷新时间
                    '''
                else:
                    GBLoadTime = math.ceil(cin/GBufBW)
                    LBufBW = 0
                    ComptTime = get_attribute(node,'latency')
                    EndTime = ReadyTime + GBLoadTime + LBufBW + ComptTime/alu + math.ceil(cout/GBufBW)
                ReadyTimeRecord.append(ReadyTime)
                EndTimeRecord.append(EndTime)
            latency.append(EndTime+FlashTime)
    elif test_type == 'cp':
        for CurFlowSeg in FlowSeg:
            pre_layer ='N' #从当前段的第一层开始
            EndTime = 0
            
            for node in CurFlowSeg:
                if node.op_type == 'Conv':
                    #FlashTime += get_attribute(node,'xb_num')[0]*get_attribute(node,'xb_num')[1]*pertime
                    k =  get_attribute(node,'kernel_shape')[0]
                    cin = get_attribute(node,'ifmsize')[1]
                    cout = get_attribute(node,'ofmsize')[1]
                    ifm = get_attribute(node,'ifmsize')[2]
                    ofm = get_attribute(node,'ofmsize')[2]

                    if pre_layer =='N':
                        ReadyTime = EndTime
                        GBLoadTime = math.ceil((ifm*(k-1)+k)*cin/GBufBW)
                        LBufBW = math.ceil(k*k*cin/BufBW )
                        ComptTime = get_attribute(node,'latency')

                        EndTime = ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime*ComptTimeScale) +  math.ceil(cout/BufBW) + math.ceil(cout/GBufBW)
                    elif pre_layer =='Conv':
                        ReadyTime = ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime*(ifm*(k-1)+k)/(ifm*ifm))*ComptTimeScale + math.ceil(cout/GBufBW)
                
                        GBLoadTime = math.ceil( cin / GBufBW) 
                        LBufBW = math.ceil(k*k*cin/BufBW )
                        ComptTime = get_attribute(node,'latency')#当前层的tc
                        EndTime = max(ReadyTime + GBLoadTime + LBufBW + ComptTime*ComptTimeScale, EndTime + math.ceil(ComptTime*(ifm/(ifm*ifm)))*ComptTimeScale)+  math.ceil(cout/BufBW) + math.ceil(cout/GBufBW)
                        
                    elif pre_layer =='MaxPool':
                        ReadyTime = ReadyTime + GBLoadTime + math.ceil(ComptTime * (ifm*(k-1)+k)/(ifm*ifm)/alu)
                        GBLoadTime = math.ceil(  cin / GBufBW) 
                        LBufBW = math.ceil(k*k*cin/BufBW )
                        ComptTime = get_attribute(node,'latency')
                        EndTime = max(ReadyTime + GBLoadTime + LBufBW + ComptTime*ComptTimeScale, EndTime + math.ceil(ComptTime*(ifm/(ifm*ifm)))*ComptTimeScale)+  math.ceil(cout/BufBW) + math.ceil(cout/GBufBW)
                        
                    pre_layer ='Conv'         
                else:
                    ReadyTime = ReadyTime + GBLoadTime + math.ceil(ComptTime*(2*2)/(ifm*ifm))*ComptTimeScale
                    GBLoadTime = math.ceil(cin/GBufBW)
                    LBufBW = 0
                    ComptTime = get_attribute(node,'latency')
                    EndTime = max(ReadyTime + LBufBW + math.ceil(ComptTime/alu), EndTime + LBufBW + 1 ) + math.ceil(cout/GBufBW)
                    pre_layer ='MaxPool'  
                ReadyTimeRecord.append(ReadyTime)
                EndTimeRecord.append(EndTime)
            latency.append(EndTime+FlashTime)
    elif test_type == 'cdcp':
        for CurFlowSeg in FlowSeg:
            pre_layer ='N' #从当前段的第一层开始
            EndTime = 0
            index = 0
            for node in CurFlowSeg:
                if node.op_type == 'Conv':
                    #FlashTime += get_attribute(node,'xb_num')[0]*get_attribute(node,'xb_num')[1]*get_attribute(node,'dup')*pertime
                    k =  get_attribute(node,'kernel_shape')[0]
                    cin = get_attribute(node,'ifmsize')[1]
                    cout = get_attribute(node,'ofmsize')[1]
                    ifm = get_attribute(node,'ifmsize')[2]
                    ofm = get_attribute(node,'ofmsize')[2]
                    
                    if pre_layer =='N':
                        ReadyTime = EndTime
                        dup =get_attribute(node,'dup')
                        GBLoadTime = math.ceil(ifm*ifm*cin/GBufBW)
                        LBufBW = math.ceil(k*k*cin/BufBW )
                        ComptTime = get_attribute(node,'latency')
                        EndTime = ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime/dup)*ComptTimeScale +  math.ceil(cout/BufBW) + math.ceil(cout*dup/GBufBW)
                    elif pre_layer =='Conv':

                        ReadyTime = ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime*(ifm*(k-1)/dup+k)/(ifm*ifm))*ComptTimeScale + math.ceil(cout/GBufBW)
                        GBLoadTime = math.ceil(cin*dup/GBufBW)
                        dup =get_attribute(node,'dup')
                        
                        LBufBW = math.ceil(k*k*cin/BufBW )
                        ComptTime = get_attribute(node,'latency')#当前层的tc
                        EndTime = max(ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime/dup)*ComptTimeScale, EndTime+ math.ceil(ComptTime*(ifm/(ifm*ifm))/dup)*ComptTimeScale) +  math.ceil(cout/BufBW) + math.ceil(cout*dup/GBufBW)

                    elif pre_layer =='MaxPool':
                        ReadyTime = ReadyTime + GBLoadTime + math.ceil(ComptTime * (ifm*(k-1)+k)/(ifm*ifm)/alu)
                        GBLoadTime = math.ceil(cin/GBufBW)
                        dup =get_attribute(node,'dup')
                        LBufBW = math.ceil(k*k*cin/BufBW )
                        ComptTime = get_attribute(node,'latency')
                        EndTime = max(ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime/dup)*ComptTimeScale, EndTime+ math.ceil(ComptTime*(ifm/(ifm*ifm))/dup)*ComptTimeScale)+  math.ceil(cout/BufBW) + math.ceil(cout*dup/GBufBW)
                    pre_layer ='Conv'        
                    index+=1
                else:
                    k =  get_attribute(node,'kernel_shape')[0]
                    cin = get_attribute(node,'ifmsize')[1]
                    cout = get_attribute(node,'ofmsize')[1]
                    ifm = get_attribute(node,'ifmsize')[2]
                    ofm = get_attribute(node,'ofmsize')[2]
                    
                    ReadyTime = ReadyTime + GBLoadTime + math.ceil(ComptTime*math.ceil(ifm * k)/(ifm*ifm)/dup)*ComptTimeScale
                    GBLoadTime = math.ceil(cin*dup/BufBW)
                    LBufBW = 0
                    ComptTime = get_attribute(node,'latency')
                    EndTime = max(ReadyTime + LBufBW + math.ceil(ComptTime/alu), EndTime + math.ceil(cout*4/alu)) + math.ceil(cout/GBufBW)
                    pre_layer ='MaxPool'  
                
                ReadyTimeRecord.append(ReadyTime)
                EndTimeRecord.append(EndTime)

            latency.append(EndTime+FlashTime)
    elif test_type == 'xdcp':
        for CurFlowSeg in FlowSeg:
            pre_layer ='N' #从当前段的第一层开始
            EndTime = 0
            for node in CurFlowSeg:
                if node.op_type == 'Conv':
                    #FlashTime += get_attribute(node,'xb_num')[0]*get_attribute(node,'xb_num')[1]*get_attribute(node,'xdup')*pertime
                    k =  get_attribute(node,'kernel_shape')[0]
                    cin = get_attribute(node,'ifmsize')[1]
                    cout = get_attribute(node,'ofmsize')[1]
                    ifm = get_attribute(node,'ifmsize')[2]
                    ofm = get_attribute(node,'ofmsize')[2]
                    
                    if pre_layer =='N':
                        ReadyTime = EndTime
                        GBLoadTime = math.ceil(ifm*ifm*cin/GBufBW)
                        dup =get_attribute(node,'dup')
                        xdup =get_attribute(node,'xdup')
                        LBufBW = math.ceil(min(k*k*cin,xbr*XBperCore)/BufBW )
                        ComptTime = get_attribute(node,'latency')
                        EndTime = ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime/xdup)*ComptTimeScale +  math.ceil(cout/BufBW) + math.ceil(cout*xdup/GBufBW)
                    elif pre_layer =='Conv':
                        ReadyTime = ReadyTime + GBLoadTime + LBufBW +  math.ceil(ComptTime*math.ceil(ifm*(k-1)/xdup+k)/(ifm*ifm))*ComptTimeScale + math.ceil(cout*xdup/GBufBW)
                        GBLoadTime = math.ceil(cin*dup/GBufBW)
                        dup =get_attribute(node,'dup')
                        xdup =get_attribute(node,'xdup')
                        
                        LBufBW = math.ceil(min(k*k*cin,xbr*XBperCore)/BufBW )
                        ComptTime = get_attribute(node,'latency')#当前层的tc
                        EndTime = max(ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime/xdup)*ComptTimeScale, EndTime+ math.ceil(ComptTime*(ifm/(ifm*ifm))/xdup)*ComptTimeScale) +  math.ceil(cout/BufBW) + math.ceil(cout*xdup/GBufBW)

                    elif pre_layer =='MaxPool':
                        ReadyTime = ReadyTime + GBLoadTime + math.ceil(ComptTime * (ifm*(k-1)+k)/(ifm*ifm)/alu)
                        GBLoadTime = math.ceil(cin*dup/GBufBW)
                        dup =get_attribute(node,'dup')
                        xdup =get_attribute(node,'xdup')
                        
                        LBufBW = math.ceil(min(k*k*cin,xbr*XBperCore)/BufBW )
                        ComptTime = get_attribute(node,'latency')
                        EndTime = max(ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime/xdup)*ComptTimeScale, EndTime+ math.ceil(ComptTime*(ifm/(ifm*ifm))/xdup)*ComptTimeScale)+  math.ceil(cout/BufBW) + math.ceil(cout*xdup/GBufBW)
                    pre_layer ='Conv'        
                    
                else:
                    k =  get_attribute(node,'kernel_shape')[0]
                    cin = get_attribute(node,'ifmsize')[1]
                    cout = get_attribute(node,'ofmsize')[1]
                    ifm = get_attribute(node,'ifmsize')[2]
                    ofm = get_attribute(node,'ofmsize')[2]
                    
                    ReadyTime = ReadyTime + GBLoadTime + math.ceil(ComptTime*math.ceil(ifm*(k-1)/xdup+k)/(ifm*ifm))*ComptTimeScale
                    GBLoadTime = math.ceil(cin*xdup/BufBW)
                    LBufBW = 0
                    ComptTime = get_attribute(node,'latency')
                    EndTime = max(ReadyTime + LBufBW + math.ceil(ComptTime/alu), EndTime + math.ceil(cout*4/alu)) + math.ceil(cout/GBufBW)
                    pre_layer ='MaxPool'  
                ReadyTimeRecord.append(ReadyTime)
                EndTimeRecord.append(EndTime)
            latency.append(EndTime+FlashTime)    
            
    elif test_type == 'xdxp':
        for CurFlowSeg in FlowSeg:
            pre_layer ='N' #从当前段的第一层开始
            EndTime = 0
            for node in CurFlowSeg:
                if node.op_type == 'Conv':
                    #FlashTime += get_attribute(node,'xb_num')[0]*get_attribute(node,'xb_num')[1]*get_attribute(node,'xdup')*pertime
                    k =  get_attribute(node,'kernel_shape')[0]
                    cin = get_attribute(node,'ifmsize')[1]
                    cout = get_attribute(node,'ofmsize')[1]
                    ifm = get_attribute(node,'ifmsize')[2]
                    ofm = get_attribute(node,'ofmsize')[2]
                    
                    if pre_layer =='N':
                        ReadyTime = EndTime
                        GBLoadTime = math.ceil(ifm*ifm*cin/GBufBW)
                        dup =get_attribute(node,'dup')
                        xdup =get_attribute(node,'xdup')
                        LBufBW = math.ceil(min(k*k*cin,xbr*XBperCore)/BufBW )
                        ComptTime = get_attribute(node,'latency')
                        EndTime = ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime/xdup)*ComptTimeScale +  math.ceil(cout/BufBW) + math.ceil(cout*xdup/GBufBW)
                    elif pre_layer =='Conv':
                        ReadyTime = ReadyTime + GBLoadTime + LBufBW +  math.ceil(ComptTime*math.ceil(ifm*(k-1)/xdup+k)/(ifm*ifm))*ComptTimeScale + math.ceil(cout*xdup/GBufBW)
                        GBLoadTime = math.ceil(min(xbr,cin)/GBufBW)
      
                        dup =get_attribute(node,'dup')
                        xdup =get_attribute(node,'xdup')
                        
                        LBufBW = math.ceil(min(k*k*cin,xbr*XBperCore)/BufBW )
                        ComptTime = get_attribute(node,'latency')#当前层的tc
                        EndTime = max(ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime/xdup)*ComptTimeScale, EndTime+ math.ceil(ComptTime*(ifm/(ifm*ifm))/xdup)*ComptTimeScale)+  math.ceil(cout/BufBW) + math.ceil(cout*xdup/GBufBW)

                    elif pre_layer =='MaxPool':
                        ReadyTime = ReadyTime + GBLoadTime + math.ceil(ComptTime * (ifm*(k-1)+k)/(ifm*ifm)/alu)
                        GBLoadTime = math.ceil(min(xbr,k*k*cin)/GBufBW)
                        dup =get_attribute(node,'dup')
                        xdup =get_attribute(node,'xdup')
                        
                        LBufBW = math.ceil(min(k*k*cin,xbr*XBperCore)/BufBW )
                        ComptTime = get_attribute(node,'latency')
                        EndTime = max(ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime/xdup)*ComptTimeScale, EndTime+ math.ceil(ComptTime*(ifm/(ifm*ifm))/xdup)*ComptTimeScale)+  math.ceil(cout/BufBW) + math.ceil(cout*xdup/GBufBW)
                    pre_layer ='Conv'  
 
                else:
                    k =  get_attribute(node,'kernel_shape')[0]
                    cin = get_attribute(node,'ifmsize')[1]
                    cout = get_attribute(node,'ofmsize')[1]
                    ifm = get_attribute(node,'ifmsize')[2]
                    ofm = get_attribute(node,'ofmsize')[2]
                    
                    ReadyTime = ReadyTime + GBLoadTime + math.ceil(ComptTime*math.ceil(ifm*(k-1)/xdup+k)/(ifm*ifm))*ComptTimeScale
                    GBLoadTime = math.ceil(cin*xdup/BufBW)
                    LBufBW = 0
                    ComptTime = get_attribute(node,'latency')
                    EndTime = max(ReadyTime + LBufBW + math.ceil(ComptTime/alu), EndTime + math.ceil(cout*4/alu)) + math.ceil(cout/GBufBW)
                    pre_layer ='MaxPool'  
                ReadyTimeRecord.append(ReadyTime)
                EndTimeRecord.append(EndTime)
            latency.append(EndTime+FlashTime)      
    elif test_type == 'xdwlp':
        for CurFlowSeg in FlowSeg:
            pre_layer ='N' #从当前段的第一层开始
            EndTime = 0
            for node in CurFlowSeg:
                if node.op_type == 'Conv':
                    #FlashTime += get_attribute(node,'xb_num')[0]*get_attribute(node,'xb_num')[1]*get_attribute(node,'xdup')*pertime
                    k =  get_attribute(node,'kernel_shape')[0]
                    cin = get_attribute(node,'ifmsize')[1]
                    cout = get_attribute(node,'ofmsize')[1]
                    ifm = get_attribute(node,'ifmsize')[2]
                    ofm = get_attribute(node,'ofmsize')[2]
                    
                    if pre_layer =='N':
                        ReadyTime = EndTime
                        GBLoadTime = math.ceil(ifm*ifm*cin/GBufBW)
                        dup =get_attribute(node,'dup')
                        xdup =get_attribute(node,'xdup')
                        LBufBW = math.ceil(min(k*k*cin,xbr*XBperCore)/BufBW )
                        ComptTime = get_attribute(node,'latency')
                        orixb = math.ceil(k*k*cin/xbr)
                        newscale = math.ceil(math.ceil(k*k*cin/orixb)/MaxR)
                        EndTime = ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime/xdup)*newscale +  math.ceil(min(xbc,cout)/BufBW) + math.ceil(min(xbc,cout)*xdup/GBufBW)
                    elif pre_layer =='Conv':
                       
                        ReadyTime = ReadyTime + GBLoadTime + LBufBW +  math.ceil(ComptTime*math.ceil(ifm*(k-1)/xdup+k)/(ifm*ifm))*newscale + math.ceil(min(xbc,cout)*xdup/GBufBW)
                        orixb = math.ceil(k*k*cin/xbr)
                        newscale = math.ceil(math.ceil(k*k*cin/orixb)/MaxR)

                        GBLoadTime = math.ceil(min(MaxR,k*k*cin)/GBufBW)
                        dup =get_attribute(node,'dup')
                        xdup =get_attribute(node,'xdup')
                        
                        delta_cycle = newscale - math.ceil(newscale/math.ceil(cin/xbc)) # 前一层的cout/xbc,可并行的机会

                        LBufBW = math.ceil(min(k*k*cin,xbr*XBperCore)/BufBW )
                        ComptTime = get_attribute(node,'latency')
                        EndTime = max(ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime/xdup)*newscale*0.95, EndTime + math.ceil(ComptTime*(ifm/(ifm*ifm))/xdup)*newscale-delta_cycle) +  math.ceil(min(xbc,cout)/BufBW) + math.ceil(min(xbc,cout)*xdup/GBufBW)

                    elif pre_layer =='MaxPool':
                        ReadyTime = ReadyTime + GBLoadTime + math.ceil(ComptTime * (ifm*(k-1)+k)/(ifm*ifm)/alu)
                        orixb = math.ceil(k*k*cin/xbr)
                        newscale = math.ceil(math.ceil(k*k*cin/orixb)/MaxR)
                        GBLoadTime = math.ceil(min(MaxR,k*k*cin)/GBufBW)
                        dup =get_attribute(node,'dup')
                        xdup =get_attribute(node,'xdup')
                        

                        math.ceil(min(k*k*cin,xbr*XBperCore)/BufBW )
                        ComptTime = get_attribute(node,'latency')
                        EndTime = max(ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime/xdup)*newscale*0.95, EndTime+ math.ceil(ComptTime*(ifm/(ifm*ifm))/xdup)*newscale)+  math.ceil(min(xbc,cout)/BufBW) + math.ceil(min(xbc,cout)*xdup/GBufBW)
                    pre_layer ='Conv'        
                else:
                    k =  get_attribute(node,'kernel_shape')[0]
                    cin = get_attribute(node,'ifmsize')[1]
                    cout = get_attribute(node,'ofmsize')[1]
                    ifm = get_attribute(node,'ifmsize')[2]
                    ofm = get_attribute(node,'ofmsize')[2]
                    
                    ReadyTime = ReadyTime + GBLoadTime + math.ceil(ComptTime*math.ceil(ifm*(k-1)/xdup+k)/(ifm*ifm))*newscale
                    GBLoadTime = math.ceil(cin*xdup/BufBW)
                    LBufBW = 0
                    ComptTime = get_attribute(node,'latency')
                    EndTime = max(ReadyTime + LBufBW + math.ceil(ComptTime/alu), EndTime + math.ceil(cout*4/alu)) + math.ceil(cout/GBufBW)
                    pre_layer ='MaxPool'  
                ReadyTimeRecord.append(ReadyTime)
                EndTimeRecord.append(EndTime)
            latency.append(EndTime+FlashTime)            
    elif test_type == 'cd':
        for CurFlowSeg in FlowSeg:
            pre_layer ='N' #从当前段的第一层开始
            EndTime = 0
            for node in CurFlowSeg:
                ReadyTime = EndTime
                if node.op_type == 'Conv':
                    #FlashTime += get_attribute(node,'xb_num')[0]*get_attribute(node,'xb_num')[1]*get_attribute(node,'dup')*pertime
                    k =  get_attribute(node,'kernel_shape')[0]
                    cin = get_attribute(node,'ifmsize')[1]
                    cout = get_attribute(node,'ofmsize')[1]
                    ifm = get_attribute(node,'ifmsize')[2]
                    ofm = get_attribute(node,'ofmsize')[2]
                    if pre_layer =='N':
                        GBLoadTime = math.ceil(ifm*ifm*cin/GBufBW)
                    else:
                        GBLoadTime = math.ceil(cin/GBufBW)

                    dup =get_attribute(node,'dup')
                    LBufBW = math.ceil(k*k*cin/BufBW )
                    ComptTime = get_attribute(node,'latency')
                    EndTime = ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime/dup)*ComptTimeScale +  math.ceil(cout/BufBW) + math.ceil(cout*dup/GBufBW)
               
                else:
                    GBLoadTime = math.ceil(cin*dup/GBufBW)
                    LBufBW = 0
                    ComptTime = get_attribute(node,'latency')
                    EndTime = ReadyTime + GBLoadTime + LBufBW + math.ceil(ComptTime/alu) + math.ceil(cout/GBufBW)
                ReadyTimeRecord.append(ReadyTime)
                EndTimeRecord.append(EndTime) 
            latency.append(EndTime+FlashTime)

    return sum(latency),ReadyTimeRecord,EndTimeRecord
