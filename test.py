from models import build_DABDETR
from datasets import build_dataset, get_coco_api_from_dataset
from util.slconfig import SLConfig
from util.visualizer import COCOVisualizer
from time import time
import torch
from testLayerNormPlugin import *
import ctypes
from cuda import cudart
import numpy as np
import os
import tensorrt as trt
# from engine import evaluate
np.random.seed(97)
num = 0
batch_size = 1


absolute_logits = [] # logits绝对误差[[平均值, 最大值, 中位数]......]
absolute_boxes = [] # boxes[[平均值, 最大值, 中位数]......]
relative_logits = [] # logits绝对误差[[平均值, 最大值, 中位数]......]
relative_boxes = [] # boxes绝对误差[[平均值, 最大值, 中位数]......]


error_num = 0

def defference(x, y, name, error_num):
    # 绝对误差
    try:
        np.testing.assert_allclose(x, y, rtol=1e-03, atol=1e-05)
    except Exception as e:
        error_num += 1
        print(e)
    absolute = np.absolute(x - y)
    mean_absolute = np.mean(absolute)
    max_absolute = max(absolute)
    median_absolute = np.median(absolute)
    print("{}绝对误差的平均值:{}, 最大值:{}, 中位数:{}".format(name, mean_absolute, max_absolute, median_absolute))
    # 相对误差
    relative = np.abs((x - y) / x)
    mean_relative = np.mean(relative)
    max_relative = max(relative)
    median_relative = np.median(relative)
    print("{}绝对误差的平均值:{}, 最大值:{}, 中位数:{}".format(name, mean_relative, max_relative, median_relative))
    
    if name == "logits":
        absolute_logits.append([mean_absolute, max_absolute, median_absolute])
        relative_logits.append([mean_relative, max_relative, median_relative])
    else:
        absolute_boxes.append([mean_absolute, max_absolute, median_absolute])
        relative_boxes.append([mean_relative, max_relative, median_relative])
    return error_num

def load_trt_model(trtFile, batch_size):
    soFilePath = "./LN_onnx/LayerNormPlugin.so"
    #soFilePath = "./oneflow_LN/LayerNormPlugin.so"
    epsilon = 1e-5
    npDataType = np.float32
    
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engineStr = f.read()
            engine = trt.Runtime(logger).deserialize_cuda_engine(engineStr)
        if engine == None:
            print("Failed loading engine!")
            exit()
        print("Succeeded loading engine!")
    else:
        print('你的engine无了')


    context = engine.create_execution_context()
    context.set_binding_shape(0, [batch_size, 3, 800, 800])
#     context.set_binding_shape(1, [batch_size, 900, 91])
#     context.set_binding_shape(2, [batch_size, 900, 4])

    print("Binding all? %s" % (["No", "Yes"][int(context.all_binding_shapes_specified)]))
    return engine, context

model_config_path = "model_zoo/DAB_DETR/R50/DAB_DETR_R50/config.json" 
model_checkpoint_path = "model_zoo/DAB_DETR/R50/DAB_DETR_R50/checkpoint.pth"
# onnx_path = "detr_sim.onnx_changed.onnx"
# img_path = "test.jpg"
trtFile = "./LN_onnx/fold_v2.plan"
#trtFile = "./oneflow_LN/fold_v2.plan"

args = SLConfig.fromfile(model_config_path)
dataset_val = build_dataset(image_set='val', args=args)
cocojs = dataset_val.coco.dataset
id2name = {item['id']: item['name'] for item in cocojs['categories']}

# load torch model
model_pth, criterion, postprocessors = build_DABDETR(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cuda:0')
model_pth.load_state_dict(checkpoint['model'])
model_pth.eval()
model_pth.cuda()
# load trt model
engine, context = load_trt_model(trtFile, batch_size)



for image, target in dataset_val:
    print("----------------------", num)
    if num == 100:
        break
    num += 1
    image = image[None]
    images = torch.tensor([])
    for i in range(batch_size):
        images = torch.cat((images, image))
#     print(image.shape)
    t1 = time()
    # pth result
    output_pth_ = model_pth(images.cuda())
    time_pth = time() - t1
    print("time_pth:", time_pth)
#     print(output_pth_)
#     output_pth = postprocessors['bbox'](output_pth_, torch.Tensor([[1.0, 1.0]]).cuda())[0]

    # trt result
#     nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
#     nOutput = engine.num_bindings - nInput
    _, stream = cudart.cudaStreamCreate()

    inputH0 = np.ascontiguousarray(images.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    outputH1 = np.empty(context.get_binding_shape(2), dtype=trt.nptype(engine.get_binding_dtype(2)))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
    _, outputD1 = cudart.cudaMallocAsync(outputH1.nbytes, stream)

    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    t2 = time()
    context.execute_async_v2([int(inputD0), int(outputD0), int(outputD1)], stream)
    time_trt = time() - t2
    print("time_trt:", time_trt)
    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaMemcpyAsync(outputH1.ctypes.data, outputD1, outputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    output_trt = {"pred_logits":torch.from_numpy(outputH0), "pred_boxes":torch.from_numpy(outputH1)}
#     output_trt = postprocessors['bbox'](output_trt, torch.Tensor([[1.0, 1.0]]))[0]
    
    # 加速倍率
    print("加速倍率:{}".format(float(time_pth)/float(time_trt)))
    # logits
    error_num = defference(output_pth_['pred_logits'].detach().cpu().numpy().flatten().astype("float"), outputH0.flatten().astype("float"), "logits", error_num)
    # boxes
    error_num = defference(output_pth_['pred_boxes'].detach().cpu().numpy().flatten().astype("float"), outputH1.flatten().astype("float"), "boxes", error_num)


    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)
    cudart.cudaFree(outputD1)

print(error_num)