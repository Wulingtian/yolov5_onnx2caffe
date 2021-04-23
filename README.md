# 环境配置

    ubuntu：18.04

    cuda：10.0

    cudnn：7.6.5

    caffe: 1.0

    OpenCV：3.4.2

    Anaconda3：5.2.0

    相关的安装包我已经放到百度云盘，可以从如下链接下载: https://pan.baidu.com/s/17bjiU4H5O36psGrHlFdM7A 密码: br7h

    cuda和cudnn的安装

    可以参考我的另一篇部署文章（TensorRT int8 量化部署 yolov5s 4.0 模型）

    Anaconda安装

    chmod +x Anaconda3-5.2.0-Linux-x86_64.sh（从上面百度云盘链接下载）

    ./Anaconda3-5.2.0-Linux-x86_64.sh

    按ENTER，然后按q调至结尾

    接受协议　yes

    安装路径　使用默认路径

    执行安装

    在使用的用户.bashrc上添加anaconda路径，比如

    export PATH=/home/willer/anaconda3/bin:$PATH

    caffe安装

    git clone https://github.com/Wulingtian/yolov5_caffe.git

    cd yolov5_caffe

    命令行输入如下内容：

    export CPLUS_INCLUDE_PATH=/home/你的用户名/anaconda3/include/python3.6m

    make all -j8

    make pycaffe -j8

    vim ~/.bashrc

    export PYTHONPATH=/home/你的用户名/yolov5_caffe/python:$PYTHONPATH

    source ~/.bashrc

# 编译过程踩过的坑

    libstdc++.so.6: version `GLIBCXX_3.4.21' not found

    解决方案：搞定 libstdc++.so.6: version `GLIBCXX_3.4.21' not found

    ImportError: No module named google.protobuf.internal

    解决方案：ImportError: No module named google.protobuf.internal

    wrap_python.hpp:50:23: fatal error: pyconfig.h: No such file or dir

    解决方案：caffe : /wrap_python.hpp:50:23: fatal error: pyconfig.h: No such file or dir

# yolov5s模型转换onnx模型

    pip安装onnx和onnx-simplifier

    pip install onnx

    pip install onnx-simplifier

    拉取yolov5官方代码

    git clone https://github.com/ultralytics/yolov5.git

    训练自己的模型步骤参考yolov5官方介绍，训练完成后我们得到了一个模型文件

    cd yolov5

    vim models/export.py 修改opset_version为10

    python models/export.py --weights 训练得到的模型权重路径 --img-size 训练图片输入尺寸

    python -m onnxsim onnx模型名称 yolov5s-simple.onnx 得到最终简化后的onnx模型

# onnx模型转换caffe模型

    git clone https://github.com/Wulingtian/yolov5_onnx2caffe.git

    cd yolov5_onnx2caffe

    vim convertCaffe.py

    设置onnx_path（上面转换得到的onnx模型），prototxt_path（caffe的prototxt保存路径），caffemodel_path（caffe的caffemodel保存路径）

    python convertCaffe.py 得到转换好的caffe模型

# caffe模型推理

    定位到yolov5_caffe目录下

    cd tools

    vim caffe_yolov5s.cpp

    设置如下参数：

    INPUT_W（模型输入宽度）

    INPUT_H（模型输入高度）

    NUM_CLASS（模型有多少个类别，例如我训练的模型是安全帽检测，只有1类，所以设置为1，不需要加背景类）

    NMS_THRESH（做非极大值抑制的阈值）

    CONF_THRESH（类别置信度）

    prototxt_path（caffe模型的prototxt路径）

    caffemodel_path（caffe模型的caffemodel路径）

    pic_path（预测图片的路径）

    定位到yolov5_caffe目录下

    make -j8

    cd build

    ./tools/caffe_yolov5s 输出平均推理时间，以及保存预测图片到当前目录下，至此，部署完成！
