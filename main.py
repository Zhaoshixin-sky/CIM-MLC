import os
import argparse
import sys
import torch
import onnx
import numpy as np
from utils import *
from configs import *
from optimization import *
from codegen import *
from model import *


def initialize(onnx_model_path, ifmsize, arch_config):
    """
    Initialize the required components for processing the ONNX model.

    Args:
    - onnx_model_path (str): Path to the ONNX model file.
    - ifmsize (tuple): Size of the input feature map.
    - arch_config (dict): Architecture configuration.

    Returns:
    - CIM (CreateCIM): Initialized CIM architecture.
    - onnx_model (onnx.ModelProto): Loaded ONNX model.
    - initializer (PreProcess): Initialized PreProcess object.
    """
    CIM = CreateCIM(arch_config)
    onnx_model = onnx.load(onnx_model_path)
    print('Model loaded successfully')

    # Calculate feature map output sizes
    ort_outs = get_layer_output(onnx_model, np.random.randn(*ifmsize).astype(np.float32))

    # Initialize PreProcess object
    initializer = PreProcess(onnx_model, ort_outs, ifmsize)

    return CIM, onnx_model, initializer


def main(onnx_model_path, ifmsize, arch_config, output_dir):
    # Initialize components
    CIM, onnx_model, initializer = initialize(onnx_model_path, ifmsize, arch_config)

    # Create output directory if not exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    '''PASS 1 Core-Wise Virtual Mapping'''
    # Set up Core config params in CIM
    core_mapping = CoreVirtualMapping(CIM)
    # Map each node based on Core hardware config (add attr to node fields)
    core_mapping.get_mapping_attributes(onnx_model, ifmsize)
    # Pass Core count in CIM for operator duplication
    core_dup = CoreWiseDuper(len(CIM.core))

    # Create network segments
    FlowSeg, DupSeg = core_dup.create_network_segments(onnx_model)

    '''PASS 2: Core-Wise Duplication.'''
    core_dup.get_corewise_dup()
    core_pipe = CoreWisePipeline(onnx_model, CIM, FlowSeg)
    core_pipe.update_dup()

    # Generate address
    addr_gen = AddrGenerator(onnx_model, CIM, initializer)
    addr_gen.AddrGen()

    # Redirect stdout to output files if output_dir is specified
    if output_dir:
        sys.stdout = open(os.path.join(output_dir, 'corewise_codegen_output.txt'), 'w')
    # Generate code
    coregen = CoreWiseCodegen(onnx_model, CIM, initializer, outinst=True)
    coregen.run()
    if ArchTem['API'] == 'Crossbar' or ArchTem['API'] == 'Wordline':
        '''PASS 3: Crossbar-Wise Duplication.'''
        cross_dup = CrossbarWiseDup(CIM, DupSeg)
        cross_dup.crossbarwise_updata_dup(ttype='real')

        '''PASS 4: Crossbar-Wise Pipeline.'''
        if output_dir:
            sys.stdout = open(os.path.join(output_dir, 'crossbarwise_codegen_output.txt'), 'w')
        crossbarWiseCodegen = CrossbarWiseCodegen(onnx_model, CIM, initializer, outinst=True)
        crossbarWiseCodegen.run()

    # Row-wise pipeline
    if ArchTem['API'] == 'Wordline':
        '''PASS 5: Wordline-Wise Pipeline.'''
        if output_dir:
            sys.stdout = open(os.path.join(output_dir, 'rcwise_codegen_output.txt'), 'w')
        row = RowWisePipeline(onnx_model, CIM)
        row.process_row_wise_pipeline()
        rcwiseCodegen = RCWiseCodegen(onnx_model, CIM, initializer, outinst=True)
        rcwiseCodegen.run()

    # Restore stdout
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CIM Model Processor")
    parser.add_argument("-onnx_model_path", type=str, help="Path to the ONNX model file")
    parser.add_argument("--ifmsize", type=int, nargs="+", default=[1, 3, 32, 32],
                        help="Size of the input feature map (default: 1 3 32 32)")
    parser.add_argument("--arch_config_module", type=str, default="configs.default_config",
                        help="Python module containing the CIM architecture configuration (default: configs.default_config)")
    parser.add_argument("--output_dir", type=str, help="Directory to save the output files")
    args = parser.parse_args()

    # Convert ifmsize to tuple
    ifmsize = tuple(args.ifmsize)

    # Import architecture configuration module
    arch_config_module = __import__(args.arch_config_module, fromlist=["ArchTem"])
    arch_config = arch_config_module.ArchTem

    # Run main function
    main(args.onnx_model_path, ifmsize, arch_config, args.output_dir)
