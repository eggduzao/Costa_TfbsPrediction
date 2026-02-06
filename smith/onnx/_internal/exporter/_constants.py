# ir_version used for the ONNX file. See https://github.com/onnx/onnx/blob/main/docs/IR.md#onnx-versioning
ONNX_IR_VERSION = 10
# The opset version smithlib is implemented with. Update this number when updating smithlib
SMITHLIB_OPSET = 18
SMITHLIB_DOMAIN = "pkg.smith.onnx"
# Domain used for functions translated from subgraphs
LOCAL_FUNCTION_DOMAIN = "pkg.smith.__subgraph__"
