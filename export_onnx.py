import torch
import torch.nn
import onnx
import os, sys
import numpy as np

from models import build_DABDETR, build_dab_deformable_detr
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops



model_config_path = "model_zoo/DAB_DETR/R50/DAB_DETR_R50/config.json" # change the path of the model config file
model_checkpoint_path = "model_zoo/DAB_DETR/R50/DAB_DETR_R50/checkpoint.pth"
args = SLConfig.fromfile(model_config_path) 
model, criterion, postprocessors = build_DABDETR(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])

model.eval()

input_names = ['inputs']
output_names = ['pred_logits', 'pred_boxes']

dummy = torch.rand(1, 3,800,800)
torch.onnx.export(
        model,dummy,
        "detr_test.onnx",
        input_names = input_names,
        output_names = output_names,
        opset_version=12,
        dynamic_axes={"inputs":{0:"batch_size"}, 
            "pred_logits":{0:"batch_size"},
            "pred_boxes":{0:"batch_size"}

            } 
    )
