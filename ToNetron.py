import onnx
from onnx import shape_inference

path = "detr_test.onnx"

save_path = "Changemodel.onnx"

onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)),save_path)
