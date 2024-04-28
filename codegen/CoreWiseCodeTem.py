# code template for core wise code generation
ConvTmpl = """CIM.read_core-conv - {ifmsize} - {Kernel} - {Stride} - {Padding} - {CoreAddrRange} - {source} - {destination}"""

MovTmpl = """Mov - {src} - {dst} - {length}"""

LinearTmpl = """CIM.read_core-fc - {ifmsize} - {CoreAddrRange} - {source} - {destination}"""

PoolTmpl = """{op_type} - {ifmsize} - {Kernel} - {Stride} - {Padding} - {source} - {destination}"""

ActivFuncTmpl = """{op_type} - {length} - {source} -{destination}"""

ALUopTmpl = """{op_type} - {input_source1} - {input_source2} - {input_length1} - {input_length2} - {destination}"""

SoftmaxTmpl = """Softmax - {length} - {source} - {destination}"""