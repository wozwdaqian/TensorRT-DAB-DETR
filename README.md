# 总述
我们将[DAB-DETR](https://github.com/IDEA-opensource/DAB-DETR)中的DAB-DETR-R50模型用tensorrt加速
# 主要工作
本团队主要工作为利用TensorRT加速模型DAB-DETR，项目开源地址：https://github.com/IDEA-opensource/DAB-DETR

使用parser来转换onnx模型生成Plan文件，在这其中使用了graphsurgeon工具修改计算图，以及使用cuda c++编写 Plugin替换myelin自动生成的Layer Normlization，然后优化kernel提升 plugin性能。

# 模型优化
本团队采用了COCO数据集作为测试集，样本数量为100张。在A10环境下，原模型单张样本的推理速度约为0.03s。我们根据[cookbook](https://github.com/NVIDIA/trt-samples-for-hackathon-cn)中的pluin和oneflow中https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/layer_norm.cuh测试了两个版本的Plugin替换Layer Normlization

- 当设定batch size为1时：
- 在使用第一版Plugin替换Layer Normlization的表现为单张推理速度约为0.01s，加速倍率在3.2-3.6倍左右。
- 使用第二版Plugin替换Layer Normlization的表现为单张推理速度约为0.01s，加速倍率在3.5-4倍左右，相对于第一个版本，加速更为稳定。

综合考虑我们使用第二个版本的plugin,代码在oneflow_LN中
# 模型输出误差
模型有logits和boxes两个输出

logits绝对误差的平均值:9.e-04, 最大值:0.002, 中位数:6.e-04；logits相对误差的平均值:1.e-04, 最大值:0.004, 中位数:7.e-05。

boxs绝对误差的平均值:1.e-04, 最大值:0.003, 中位数:7.e-05；logits相对误差的平均值:6.e-04, 最大值:0.02, 中位数:2.e-04。

## 项目环境搭建

1. 拉取镜像

        docker pull nvcr.io/nvidia/pytorch:21.12-py3
        


2. 运行容器

    如果你的docker是19.03后者更新的版本，则运行:

        docker run --gpus all -it -v local_dir:container_dir nvcr.io/nvidia/pytorch:21.12-py3


    如果是19.02或者更早的版本，则运行：

        nvidia-docker run -it --rm -v local_dir:container_dir nvcr.io/nvidia/pytorch:21.12-py3

3. 安装tensor8.4版本，本团队使用的版本为8.4.1.5，官网链接：https://developer.nvidia.com/nvidia-tensorrt-download

   下载下来的tensorrt压缩包为TensorRT-8.4.1.5.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz

    这里推荐安装在/opt目录下，否则在替换Plugin时需要修改makefile文件中的TensorRT路径。然后解压这个文件。


        cd /opt
        tar -zxvf TensorRT-8.4.1.5.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz


    添见环境变量

        sudo vi ~/.bashrc
        export LD_LIBRARY_PATH=/opt/TensorRT-8_4_1_5/lib:$LD_LIBRARY_PATH
        source ~/.bashrc


4. 安装tensorrt的python包。进入tensorrt压缩之后的文件夹，然后进入里面的python文件夹

        cd /opt/TensorRT-8_4_1_5/python


    python文件夹里面有很多版本，使用pip安装自己对应的python版本，例如本团队的是python3.8，则执行

        pip install tensorrt-8.4.1.5-cp38-none-linux_x86_64.whl

    安装 uff 包

        cd ../uff    
        pip install uff-0.6.9-py2.py3-none-any.whl

    安装 graphsurgen 包

        cd ../graphsurgeon     
        pip install graphsurgeon-0.4.6-py2.py3-none-any.whl

    安装 onnx_graphsurgeon 包

        cd ../onnx_graphsurgeon  
        pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl

5. 进入项目文件夹，下载项目所需依赖库，推荐使用国内的源进行下载，否则下载速度太慢

        pip install -r requirements.txt
        pip install onnxruntime
        pip install onnx
        pip install cuda-python
        pip install colored
6. 编译CUDA计算器

        cd models/dab_deformable_detr/ops/
        python setup.py build install

7. 测试依赖库是否装齐（若显存过小，会报错out of memory，不需要管他）

        python test.py
        cd ../../..

# 项目优化

下载原模型到本地目录下的/model_zoo/DAB_DETR/R50/DAB_DETR_R50/，若因为模型太大无法拉取，可从源项目下载，开源地址为https://cloud.tsinghua.edu.cn/d/3aa9d0da60e8423dab54/?p=%2FDAB_DETR%2FR50&mode=list，后将模型config中"coco path"改为"COCODIR/","aux_loss"改为false。

本项目的测试样本为coco数据集，路径如下


TensorRT-DAB-DETR
|---------COCODIR

          |-----COCO2017
          
                |---annotations
                
                |---train2017
                
                |---val2017


将模型导出为onnx

        python export_onnx_sim.py

执行优化过程，常量折叠

        polygraphy surgeon sanitize detr_sim_changed.onnx --fold-constants -o fold_v3.onnx

手动书写Plugin替换Layer Normlization
1. 第一版Plugin

        cd LN_onnx
        python main.py

2. 第二版Plugin

        cd oneflow_LN
        python main.py

# 精度测试

测试基础版Plugin或者oneflow版本的Plugin需要分别修改test.py文件中55和87行，修改Plan文件与so文件的路径

        python test.py

# 原始模型

## 模型简介

该模型适用于目标检测领域，其直接使用框坐标作为 Transformer 的decoders中的查询，并逐层动态更新它们。使用框坐标不仅有助于消除 DETR 中缓慢的训练收敛问题，而且还允许使用框的宽度和高度信息来调制attention。结果，它在相同设置下的类似 DETR 的检测模型中导致 MS-COCO 基准测试的最佳性能，例如，使用 ResNet50-DC5 作为主干训练 50 个 epoch 的 AP 45.7%。

本模型的框架见struction.jpg。

## 模型优化的难点

该项目在训练后，模型和参数是分开保存的，因此在导入时，应分别导入模型以及参数后再进行导出onnx操作。再使用parser转换onnx模型生成Plan文件时，模型的Pad节点不被TensorRT8.2支持，且TensorRT-8.4.1.4不支持ND ShapeTensor。

## 优化过程

参照着开源项目的推理脚本，导入训练好的模型和参数，之后再转成onnx模型，简便起见，我们再torch.onnx.export方法中只将batch_size设定为动态尺寸，其余全部固定，并令opset_version=12。

在导出onnx时，本团队使用了onnxruntime来检测了导出精度，确保了与原模型的精度差在符合的要求内,之后利用onnxsim优化导出的onnx模型。。

在利用netron查看导出的onnx文件是，无法观察到每个输入输出张量的尺寸及数据类型，因此，利用onnx中shape_inference另存为onnx，使之能在netron中显示每个张量的尺寸及数据类型。

在利用导出的onnx直接转成Plan文件时，显示报错信息：

        [TRT][E][shufflelodecpp:ssymbolicExecute::392]ErrorCode4:Internal Error(Reshape_33 : Shufflelayer applied to shape tensor must have or i reshape dinensions : dinensions were (-12]) Failed parsig ONNx file!
        In node 44 (parseGraph]:INVALID NODE:Invalid Node-Pad 44
        TshuffleNode.coo:isvmbolicExecute::3921ErconCode 4:Internal Error(Reshane_33: IShufflelaveranolied to shane tensor must have 0 or 1 reshane dimensions: d imensions were[-1 21)

经查询可得：TensorRT-8.4.1.4不支持ND ShapeTensor，Reshape_33 [-1, 2] -1表示自动推导，当前版本8.4.1.4还不支持。

之后利用polygraphy进行优化，利用常量折叠优化部分算子：

        polygraphy surgeon sanitize ChangeDETR.onnx --fold-constants -o fold.onnx

执行后通过netron打开优化好的flod.onnx，发现带有自动推导的Reshape节点被自动折叠优化。查询后，发现官方的回答如下：

        Hi, I think padding related node is causing error, we don’t support 2D shape tensors yet. We can try workaround constant-fold with polygraphy. After this we are able to successfully generate engine. Please try polygraphy surgeon sanitize --fold-constants grid_sample.onnx -o 2001833/folded.onnx For more details, Thank you.

个人理解就是虽然当前的trt不支持该算子，但是可以利用提供了polygraphy来fold该算子，以此实现引擎的构建。

Layer Normlization是深度学习模型中较为常见的操作之一，其 CUDA Kernel 实现的高效性会影响很多网络最终的训练速度，Softmax 这种优化方法也适用于 LayerNorm，LayerNorm 的数据也可以表示为 (num_rows, num_cols)，计算过程中对每一行的元素做 Reduce 操作求均值方差。因此我们使用了和 Softmax 同样的优化方法来优化 LayerNorm 操作。

但是在不同的训练框架和API组合导出成onnx文件的时候，Layer Normlization的形式是不统一的，所以TensorRT无法自动融合。本模型的Layer Normlization的形式在netron中表示为从ReduceMean到Add共有9个节点组成，若将其9个节点合并成为1个Plugin，将加快模型的推理速度。

经过查询，发现该模型中共有35个Layer Normlization，若将这一部分替换为自己手动书写的Plugin，将会大大加快推理的速度。

以 PyTorch 为例，LayerNorm 的接口为:

        torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine = True , device=None, dtype=None)

第一个参数 normalized_shape 只能是输入 x_shape 的后几维，例如 x_shape 为 (N, C, H, W), normalized_shape 可以是 (W)， (H, W) ，(C, H, W) 或 (N, C, H, W)。输入 x 在 normalized_shape 这几维上求均值和方差。

第三个参数 elementwise_affine 代表是否要对 normalize 的结果做变换，即 normalize 的结果乘 gamma，加 beta。若 elementwise_affine=True，就多了两个模型参数 gamma 和beta，形状为 normalized_shape。

例如对于输入 x 形状为 (N, C, H, W)， normalized_shape 为 (H, W) 的情况，可以理解为输入 x 为 (N*C, H*W)，在 N*C 个行上，每行有 H*W 个元素，对每行的元素求均值和方差，得到 N*C 个 mean 和 inv_variance，再对输入按如下 LayerNorm 的计算公式计算得到 y。若 elementwise_affine=True ，则有 H*W 个 gamma 和 beta，对每行 H*W 个的元素做变换。

本团队一共优化了两版的Layer Normlization的Plugin。

首先是需要定位到所有的Layer Normlization所在位置：

Layer Normlization模块中Div算子的上一个输入节点即为Sub，因此通过代码if node.op == 'Div'and node.i(0).op == 'Sub'来定位该节点的位置。通过onnx_graphsurgeon来修改计算图，添加新的LayerNorm节点与他的输出，由计算图可以看出，LayerNorm节点的输入为sub和ReduceMean的输入，即为Div的上两级节点的输入，通过node.i(0).i(0).outputs[0]表示，最后设置LayerNorm节点属性"epsilon为[1.e-05]。

第一版的项目开源地址为：https://github.com/NVIDIA/trt-samples-for-hackathon-cn/cookbook/06-PluginAndParser/pyTorch-LayerNorm

其kernel的实现基本与串行代码一致，首先计算好所有元素的均值和方差，再对所有元素进行一个线性调整后写出，但是只接受float32类型的输入和输出，并且其只能对256个元素进行Layer Normlization的计算，如果元素超过256，则需要见 PluginRepository。

第二版的项目开源地址为：https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/layer_norm.cuh。学习了该博主的优化方式，原文链接：https://blog.csdn.net/oneflow_official/article/details/121974648

和 Softmax 一样，LayerNorm 也采用分段函数优化，对于不同的 num_cols 范围，采用不同的实现，以在各种情况下都能达到较高的有效带宽。

在每种实现中都采用了一个公共的优化：向量化访存.通过向量化内存操作来提高 CUDA Kernel 性能，很多 CUDA Kernel 都是带宽受限的，使用向量化内存操作可以减少总的指令数，减少延迟，提高带宽利用率。

理论上来说，在计算 LayerNorm 的过程中，输入 x 需要被读两次，第一次用于计算均值和方差。第二次用于得到均值和方差后的计算过程。而对 Global Memory 的访问操作是昂贵的，如果能将输入 x 先存起来，不重复读，就可以提升性能。在 GPU 中将输入 x 存起来可以使用寄存器或 Shared memory，但是寄存器资源和 Shared memory 资源都是有限的，如果 num_cols 过大，就会超出资源的使用限制，因此我们针对不同 num_cols 采用不同的实现，下面分别进行介绍：

1. num_cols <= 1024的情况

针对 num_cols <= 1024 的情况，以 Warp 为单位处理一行或两行，将输入 x 存储到寄存器中。

硬件上并行执行的32个线程称之为一个 Warp，同一个 Warp 的32个 thread 执行同一条指令， Warp是 GPU 调度执行的基本单元。线程块和元素的对应关系如上图所示，每个 Warp 的 threads 处理一行元素，每个 block 有 block_size / warp_size 个 Warp，每个 block 处理 block_size / warp_size 行元素。

具体的处理流程是，如下图所示，每行有 num_cols 个元素，每个 warp 处理一行，因此每个线程需要处理 num_cols / warp_size 个元素，每个线程读取自己需要处理的元素存储到寄存器中，并用 Welford 算法计算好均值和方差后，Warp 中的所有线程执行一次 WelfordWarpAllReduce，这样每个线程上就得到了正确的均值和方差参与后续计算。

在这里有个模板参数 thread_group_width，当 num_cols > pack_size * WarpSize 时，thread_group_width 为 WarpSize。当 num_cols 太小，即 num_cols<pack_size * WarpSize 时，一个 Warp 内的线程不是全部处理有效的值，此时我们采用更小的thread_group_width，取值可能是16、8、4、2、1，由 num_cols 决定，并且每个线程处理两行增加并行度。

此外，在读写输入输出时，我们采用向量化访存的优化，在满足条件时，将 pack_size 个元素 pack 成更大的数据类型读入，下图为 pack_size=2 时的示意图，每个线程以更大的数据类型读入元素，可以更好的利用显存带宽。

LayerNormWarpImpl 的实现的模板参数的意义分别如下：

LOAD、STORE 分别代表输入输出，使用load.template load<pack_size>(ptr, row_id, col_id); 和 store.template store<pack_size>(ptr, row_id, col_id); 进行读取和写入。使用 LOAD 和 STORE 有两个好处：a) 可以在 CUDA Kernel中只关心计算类型 ComputeType，而不用关心具体的数据类型 T。b) 只需要加几行代码就可以快速支持 LayerNorm 和其他 Kernel Fuse，减少带宽需求，提升整体性能。

ComputeType 代表计算类型。pack_size 代表向量化访存操作的 pack 元素的个数，我们将几个元素 pack 起来读写，提升带宽利用率。

cols_per_thread 代表每个线程处理的元素个数。

thread_group_width 代表处理元素的线程组的宽度，当 cols > pack_size * warp_size 时，thread_group_width 就是warp_size，即32。当 cols < pack_size * warp_size 时，就根据 cols 大小用 1/2个warp 或 1/4个warp 来处理每行的元素。采用更小的 thread_group_width 后，WarpAllReduce需要执行的轮次也相应减少。

rows_per_access 代表每个 thread_group 一次处理的行数，当 cols 较小且 thread_group_width 小于warp_size时，若 rows 能被2整除，我们就让每个线程处理2行来增加指令并行度，从而提升性能。

padding 代表当前是否做了 padding，若 cols 不是 warp_size 的整数倍，我们会把它padding 到最近的整数倍处理。

2. num_cols > 1024的情况

针对 num_cols > 1024 ，以 block 为单位处理一行，利用 Shared Memory 存储输入数据

对于 num_cols > 1024 的情况，每个 block 处理一行元素，将输入 x 存储到 Shared Memory中。

具体的处理流程是，如下图所示，每行有 num_cols 个元素，每个 block 处理一行，因此每个线程需要处理 num_cols / block_size 个元素，每个线程读取自己需要处理的元素存储到 Shared Memory 中，并用 Welford 算法计算好均值和方差后，block 中的所有线程执行一次WelfordBlockAllReduce，这样每个线程上就得到了正确的均值和方差参与后续计算。

WelfordBlockAllReduce 是借助 WelfordWarpReduce 操作完成的，具体逻辑是，一个 Block 中最多有32个 Warp，对所有的 Warp 先执行一次 WelfordWarpReduce，执行完后，每个 warp 中的第一个线程，即 lane_id=0 的线程上得到当前 WelfordWarpReduce 的结果，再将每个 Warp 的第一个线程的结果拷贝到一块 Shared Memory buffer 中，再用第一个 Warp 的32个线程执行一次 WelfordWarpReduce，此时第一个 Warp 中的 lane_id=0 的线程上得到的就是 block 中所有线程reduce 的结果。再借助 Shared Memory，将该结果 broadcast 到 block 中的所有线程上，即完成了 WelfordBlockAllReduce 的操作。

值得注意的是 GPU 上 Shared Memory 资源同样有限，当 num_cols 超过一定范围时需要占用的Shared Memory 可能就超出了最大限制，Kernel 就无法启动起来。

因此，我们采用 cudaOccupancyMaxActiveBlocksPerMultiprocessor 函数判断当前硬件资源条件下 Kernel 是否能成功启动，仅在返回值大于0时采用这种方案。

此外，由于 Block 内线程要做同步，当 SM 中正在调度执行的一个 Block 到达同步点时，SM 内可执行 Warp 逐渐减少，若同时执行的 Block 只有一个，则 SM 中可同时执行的 Warp 会在此时逐渐降成0，会导致计算资源空闲，造成浪费，若此时同时有其他 Block 在执行，则在一个 Block 到达同步点时仍然有其他 Block 可以执行。

当 block_size 越小时，SM 可同时调度的 Block 越多，因此在这种情况下 block_size 越小越好。但是当在调大 block_size，SM 能同时调度的 Block 数不变的情况下，block_size 应该是越大越好，越大就有越好的并行度。因此代码中在选择 block_size 时，对不同 block_size 都计算了 cudaOccupancyMaxActiveBlocksPerMultiprocessor，若结果相同，使用较大的 block_size。

在构建完毕引擎后，利用nsight systems分析一次构建和运行期的时间，图片详见nsys1-4.png。

## 精度与加速效果

测试的脚本为./test/test_engine.py,将两个版本生成的Plan文件添加到./test/路径下，再修改test——engine.py中engine的路径即可进行模型的精度与速度的优化对比。

我们输入的数据是用COCO的val里的图片，变量num表示检测的图片数量,本样本数量为100张。

当batch size为1时：

本团队采用了COCO数据集作为测试集，样本数量为100张。原模型单张样本的推理速度约为0.03s。


在使用第一版Plugin替换Layer Normlization的表现的加速倍率在1.43倍左右。

使用第二版Plugin替换Layer Normlization的表现的加速倍率在1.43倍左右。

logits绝对误差的平均值:9.e-04, 最大值:0.002, 中位数:6.e-04；logits相对误差的平均值:1.e-04, 最大值:0.004, 中位数:7.e-05。

boxs绝对误差的平均值:1.e-04, 最大值:0.003, 中位数:7.e-05；logits相对误差的平均值:6.e-04, 最大值:0.02, 中位数:2.e-04。

# Bug报告
## Environment
- TensorRT 8.4 GA
- NGC镜像nvcr.io/nvidia/pytorch:21.12-py3
- NVIDIA driver version：470
## Reproduction Steps
- 在导入模型前将原模型的config.json文件中aux_loss改为true，按照readme中步骤开始导入onnx模型直至生成engine。之后运行当前路径下的test.py。
## Expected Behavior
- Provide 导入engine成功并开始执行推理。
## Actual Behavior
- context.execute_async_v2([int(inputD0), int(outputD0), int(outputD1)], stream) 处报错。
- 报错信息：[TRT] [E]1:[cudaDriverHelpers.cpp;:operator()::29] Error Code 1: Cuda Driver (misaligned address)
## Additional Notes
- config.json文件中aux_loss改为fales时即可成功执行推理过程。
