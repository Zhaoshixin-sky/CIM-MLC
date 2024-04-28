# code template for crossbar wise code generation

ConvTmpl = """CIM.read_xb - {xbaddr}"""

LinearTmpl = """
"""

PoolTmpl = """{op_type} - {ifmsize} - {Kernel} - {Stride} - {Padding} - {source} - {destination}"""

ActivFuncTmpl = """{op_type} - {length} - {source} -{destination}"""

ALUopTmpl = """{op_type} - {input_source1} - {input_source2} - {input_length1} - {input_length2} - {destination}"""

MovTmpl = """load - {source} - {destination} - {length}"""

PadTmpl = """pad - {leftpad} - {source} - pad - {rightpad}"""

SoftmaxTmpl = """softmax - {length} - {source} - {destination}"""