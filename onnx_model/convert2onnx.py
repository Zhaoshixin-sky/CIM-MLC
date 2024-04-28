import torch

def torch2onnx(model,input,name):
    model.eval() 
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model,
                      input,
                      str(name)+'.onnx',
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes={'input' : {0: 'batch'},  
                                    'output' : {0: 'batch'}})
    
