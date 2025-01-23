DL-framework开发目标:
重新梳理：
- 当前状况: 构建计算图的时候，每次都需要进行链接，以为着必须要使用动态指针管理，但每次使用动态指针来声明进行操作不太人性化
- 解决方案: 模仿libtorch的解决方案，构建一个tensor_impl来实现底层，而tensor控制的则是指向tensor_impl的指针
  - 详细见：https://github.com/pytorch/pytorch/blob/d9507548d83cef3223159b354f7896cead836931/aten/src/ATen/templates/TensorBody.h#L629
TensorBase: https://github.com/pytorch/pytorch/blob/c09bf71bd61f2c7786ad7c8d5d7c09b9d619b06e/aten/src/ATen/core/TensorBase.h


问题2：当我在对Tensor进行运算的时候，它需要构建一个result tensor，但当函数结束以后，该tensor的生命周期也结束了

由于tensor的内部是仍然是shared_ptr，意味着只要通过shallow copy，仍然可以使用当前代码，返回值就是plain tensor
这就使tensor返回出去以后，得到的是不同的tensor对象，但仍然能够保持内部数据的一致性。
但问题就是该方案只允许每个tensor同时仅有一个owner，否则多个地方对非tensor核心数据(storage)的修改，无法同步。
而为了同步，我需要将tensor所有的内部封装成一个基础对象，并使用shared_ptr绑定，这样进行修改的时候，就是对整体进行修改，而不是局部某个数据

内部的层名字叫做TensorMeta

为训练的时候创建pipeline
fetch data -> train -> validate -> log

结构是去区分layer and module
layer用来表示每个函数，而module用来表示大的，可以优化的部分
module让其可以自定义

1. 构建计算图
2. 构建推理引擎
   1. 对于tensor而言需要构建两套系统，一套用来在运算的时候构建图，一套用来执行推理
3. 构建训练引擎
4. 构建日志系统
   1. 普通的日志板块：输出训练数据，每个epoch中的相关信息，计算出来
   2. 绘图模式：将训练过程用进度条来表示
5. SIMD加速
   OpenBlas Library or intel mkl
6. CUDA加速
7. 实现对训练集的封装

#### TensorIterator
1. 可以添加，但不是最重要的模块
2. 添加的意义第一点在于在做一些元素操作的时候，我可以不用改变内存布局，直接使用iterator来访问 (等于改变stride)
3. 第二点在于允许我使用不同的内存布局 (row-major form, column-major form)

新的深度学习框架命名：Axon++; Cortex++; TensorLite; 


#### 广播机制
1. 对于scalar，系统不会进行广播，而对于非scalar，且shape doesn't match，则会使用广播进行操作
   1. e.g. +-*/和pow运算都遵从该法则
2. 广播会交给tensor层进行操作，而并非底层的kernel进行实现
3. kernel只会负责scalar和相同shape的两个分支的底层实现

#### 计算图上的问题
1. 对于多阶求导，我是否需要在每个layer反向传播的时候，将其detach出来


About dtype conversion to template:
使用dtype的目的在于如果使用template，用户在每次调用相关工具时，都需要重新specify使用的类型
这种方法可行的前提在于类型的consistency，如果没有连续性，或者连续性对用户隐藏，则用户可能会在类型上出错
最好的情况是完全不用模板，所有内容全部使用dtype，而背后将其转换为template

Libtorch notes:
libtorch底层使用的dtype是一个struct
包含了：
1. Code
2. bits

使用register注册表来实现类型转换
一个方法是基于类的注册
一个方法是基于函数的注册

写一个struct叫做dtype
dtype里面包含了每种类型
已经dtype_enum

Layer层只会负责我的计算部分
而后续的params存储之类的，一概不负责

注意：广播操作有问题，在矩阵乘法当中，无法复制matrix

构建一个模型类，继承自BaseModule
拥有一个vector，放着的是其他的所有baseModule
另一个baseModule，放着的是损失函数，
我需要每次构建一个model的实话，前面的内容使用
make_model({LinearLayer, ActivationLayer,...}, Loss)
model.backward() -> loss.backward() + loss.graph->clear_grad;

而与此同时，我也允许用户使用函数式变成来实现

处理batch_size
了解输入的是包含batch，还是不包含batch，否则在mse输出的内容


今天去测试boston housing price prediction
需要重新构建整个layer的方法，因为现在的状态是每次计算，都要重新构建计算图，我需要构建完一次以后，就完成了，因此我需要给每个module添加一个forward函数
然后他们拥有baseLayer，如果是第一次，就构建计算图，而后续直接传入状态，状态作为一个enum或者uint8，然后可以有训练模式，动态模式
然后直接去走计算本身，然后修改输出tensor的值，然后继续到下一个layer


https://levelup.gitconnected.com/tensor-programming-in-c-custom-reducers-ac9420402cab