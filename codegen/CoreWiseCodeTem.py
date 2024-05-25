# code template for core wise code generation
ConvTmpl = """CIM.read_core-conv - {ifmsize} - {Kernel} - {Stride} - {Padding} - {CoreAddrRange} - {source} - {destination}"""

MovTmpl = """Mov - {src} - {dst} - {length}"""

LinearTmpl = """CIM.read_core-fc - {ifmsize} - {CoreAddrRange} - {source} - {destination}"""

PoolTmpl = """{op_type} - {ifmsize} - {Kernel} - {Stride} - {Padding} - {source} - {destination}"""

ActivFuncTmpl = """{op_type} - {length} - {source} -{destination}"""

ALUopTmpl = """{op_type} - {input_source1} - {input_source2} - {input_length1} - {input_length2} - {destination}"""

SoftmaxTmpl = """Softmax - {length} - {source} - {destination}"""

TransposeTmpl = """Transpose - {input_source} - {input_length} - {ifmsize} - {perm} - {ofmsize} - {destination}"""

ReshapeTmpl = """{op_type} - {input_source} - {input_length} - {ifmsize} - {ofmsize} - {destination}"""

GatherTmpl = """Gather - {input_source} - {input_length} - {ifmsize} - {indices} - {axis} - {ofmsize} - {destination}"""

SliceTmpl = """Slice - {input_source} - {input_length} - {ifmsize} - {starts} - {ends} - {axes} - {steps} - {ofmsize} - {destination}"""

ExpandTmpl = """Expand - {input_source} - {input_length} - {ifmsize} - {ofmsize} - {destination}"""

UnsqueezeTmpl = """Unsqueeze - {input_source} - {input_length} - {ifmsize} - {axes} - {ofmsize} - {destination}"""

WhereTmpl = """Where - {input_source1} - {input_source2} - {input_source3} - {input_length1} - {input_length2} - {input_length3} - {ofmsize}- {destination}"""

ElementWiseTmpl = """{op_type} - {input_source} - {input_length} - {ifmsize} - {ofmsize} - {destination}"""

ReduceMeanTmpl = """ReduceMean - {input_source} - {input_length} - {ifmsize} - {axes} - {ofmsize} - {destination}"""

FlattenTmpl = """Flatten - {input_source} - {input_length} - {ifmsize} - {axis} - {ofmsize} - {destination}"""
