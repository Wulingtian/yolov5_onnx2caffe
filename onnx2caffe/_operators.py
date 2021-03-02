from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe import params as P
import math
import numpy as np
from ._graph import Node, Graph
from MyCaffe import Function as myf


def _compare(a, b, encoding="utf8"):  # type: (Text, Text, Text) -> bool
    if isinstance(a, bytes):
        a = a.decode(encoding)
    if isinstance(b, bytes):
        b = b.decode(encoding)
    return a == b


def make_input(input):
    name = input[0]
    output = input[0]
    output = [output]
    shape = input[2]
    shape = list(shape)
    input_layer = myf("Input", name, [], output, input_param=dict(shape=dict(dim=shape)))
    return input_layer


def _convert_conv(node, graph, err):
    weight_name = node.inputs[1]  # 0 是上层节点名，1 开始才是本层参数 name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    node_name = node.name
    W = None
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name, ))
    is_deconv = False
    if node.op_type.endswith("Transpose"):
        is_deconv = True
    bias_flag = False
    bias = None
    if len(node.inputs) > 2:
        bias = node.input_tensors[node.inputs[2]]
        bias_flag = True
    dilations = node.attrs.get("dilations", [1, 1])
    # groups = 1
    groups = node.attrs.get("group", 1)
    kernel_shape = node.attrs["kernel_shape"]
    pads = node.attrs.get("pads", [0, 0, 0, 0])
    strides = node.attrs["strides"]

    layer = myf("Convolution", node_name, [input_name], [output_name],
                kernel_h=kernel_shape[0], kernel_w=kernel_shape[1],
                stride_h=strides[0], stride_w=strides[1], group=groups,
                pad_h=pads[0], pad_w=pads[1],
                num_output=W.shape[0], dilation=dilations[0], bias_term=bias_flag)

    graph.channel_dims[output_name] = W.shape[0]
    return layer


def _convert_relu(node, graph, err):
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    name = str(node.name)

    if input_name == output_name:
        inplace = True
    else:
        inplace = False

    layer = myf("ReLU", name, [input_name], [output_name], in_place=inplace)
    # l_top_relu1 = L.ReLU(l_bottom, name=name, in_place=True)

    graph.channel_dims[output_name] = graph.channel_dims[input_name]

    return layer

def _convert_leaky_relu(node,graph,err):
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    name = str(node.name)
    alpha = node.attrs.get("alpha", 1)

    print(node.attrs["alpha"])

    if input_name==output_name:
        inplace = True
    else:
        inplace = False

    layer = myf("ReLU",name,[input_name],[output_name],in_place=inplace, negative_slope=alpha)
    # l_top_relu1 = L.ReLU(l_bottom, name=name, in_place=True)

    graph.channel_dims[output_name] = graph.channel_dims[input_name]

    return layer


def _convert_prelu(node, graph, err):
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    name = str(node.name)

    if input_name == output_name:
        inplace = True
    else:
        inplace = False

    layer = myf("PReLU", name, [input_name], [output_name], in_place=inplace)
    # l_top_relu1 = L.ReLU(l_bottom, name=name, in_place=True)

    graph.channel_dims[output_name] = graph.channel_dims[input_name]

    return layer

def _convert_sigmoid(node, graph, err):
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    name = str(node.name)

    if input_name == output_name:
        inplace = True
    else:
        inplace = False

    layer = myf("Sigmoid", name, [input_name], [output_name], in_place=inplace)
    # l_top_relu1 = L.ReLU(l_bottom, name=name, in_place=True)

    graph.channel_dims[output_name] = graph.channel_dims[input_name]

    return layer


def _convert_BatchNorm(node, graph, err):
    epsilon = node.attrs.get("epsilon", 1e-5)
    scale = node.input_tensors[node.inputs[1]]
    bias = node.input_tensors[node.inputs[2]]
    mean = node.input_tensors[node.inputs[3]]
    var = node.input_tensors[node.inputs[4]]
    node_name = node.name

    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])

    if input_name == output_name:
        inplace = True
    else:
        inplace = False

    bn_layer = myf("BatchNorm", node_name + "_bn", [input_name], [output_name], eps=epsilon, use_global_stats=True, in_place=inplace)
    scale_layer = myf("Scale", node_name, [output_name], [output_name], in_place=True, bias_term=True)

    graph.channel_dims[output_name] = graph.channel_dims[input_name]

    return bn_layer, scale_layer


def _convert_Add(node, graph, err):
    input_name_list = [str(i) for i in node.inputs]
    output_name = str(node.outputs[0])
    node_name = node.name

    max_dim = 0
    for name in input_name_list:
        print(graph.channel_dims)
        #if graph.channel_dims[name] > max_dim:
           # max_dim = graph.channel_dims[name]

    if 'broadcast' in node.attrs:
        if node.attrs['broadcast'] == 1:
            input_node_number = len(input_name_list)
            if input_node_number != 2:
                return err.unsupported_op_configuration(node, "Broadcast Add must has 2 input, not {}".format(input_node_number))
            axis = node.attrs['axis']
            flat_layer = myf("Flatten", node_name + '_flat', [input_name_list[1]], [output_name + '_flat'])
            layer = myf("Bias", node_name, [input_name_list[0], output_name + '_flat'], [output_name], axis=axis)
            # layer = myf("Bias", node_name, input_name_list, [output_name], bias_term = False, axis = axis)
            graph.channel_dims[output_name] = graph.channel_dims[input_name_list[0]]
            return flat_layer, layer
    #print("**********************************")
    #print(input_name_list)
    #print("**********************************")
    layer = myf("Eltwise", node_name, input_name_list, [output_name], operation=P.Eltwise.SUM)
    graph.channel_dims[output_name] = graph.channel_dims[input_name_list[0]]
    return layer


def _convert_Mul(node, graph, err):
    input_name_list = [str(i) for i in node.inputs]
    output_name = str(node.outputs[0])
    node_name = node.name

    # max_dim = 0
    # for name in input_name_list:
    #     if graph.channel_dims[name]>max_dim:
    #         max_dim = graph.channel_dims[name]

    if 'broadcast' in node.attrs:
        if node.attrs['broadcast'] == 1:
            input_node_number = len(input_name_list)
            if input_node_number != 2:
                return err.unsupported_op_configuration(node, "Broadcast Mul must has 2 input, not {}".format(input_node_number))
            axis = node.attrs['axis']
            flat_layer = myf("Flatten", node_name + '_flat', [input_name_list[1]], [output_name + '_flat'])
            layer = myf("Scale", node_name, [input_name_list[0], output_name + '_flat'], [output_name], bias_term=False, axis=axis)
            graph.channel_dims[output_name] = graph.channel_dims[input_name_list[0]]
            return flat_layer, layer

    layer = myf("Eltwise", node_name, input_name_list, [output_name], operation=P.Eltwise.PROD)
    graph.channel_dims[output_name] = graph.channel_dims[input_name_list[0]]
    return layer


def _convert_Reshape(node, graph, err):
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    if len(node.inputs) == 1:
        shape = tuple(node.attrs.get('shape', ()))
    else:
        shape = tuple(node.input_tensors[node.inputs[1]])
    # if shape == ():

    if input_name == output_name:
        inplace = True
    else:
        inplace = False
    if len(shape) == 2:
        layer = myf("Flatten", node_name, [input_name], [output_name], in_place=inplace)
        graph.channel_dims[output_name] = shape[1]
        return layer
    elif len(shape) == 4 or len(shape) == 3 or len(shape) == 5:
        graph.channel_dims[output_name] = shape[1]
        layer = myf("Reshape", node_name, [input_name], [output_name], reshape_param=dict(shape=dict(dim=list(shape))))
        return layer
    else:
        return err.unsupported_op_configuration(node, "Reshape dimention number shall be 2 or 4")


def _convert_Flatten(node, graph, err):
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    # shape = tuple(node.attrs.get('shape', ()))
    if input_name == output_name:
        inplace = True
    else:
        inplace = False
    layer = myf("Flatten", node_name, [input_name], [output_name], in_place=inplace)
    # graph.channel_dims[output_name] = shape[1]
    return layer


def _convert_Permute(node, graph, err):
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    if len(node.inputs) == 1:
        shape = tuple(node.attrs.get('perm', ()))
    else:
        shape = tuple(node.input_tensors[node.inputs[1]])

    if len(shape) == 3 or len(shape) == 4 or len(shape) == 5:
        layer = myf("Permute", node_name, [input_name], [output_name], permute_param=dict(order=list(shape)))
        return layer
    else:
        return err.unsupported_op_configuration(node, "Reshape dimention number shall be 2 or 4")

def _convert_Softmax(node, graph, err):
    node_name = node.name
    input_name_list = [str(i) for i in node.inputs]
    output_name = str(node.outputs[0])
    axis = node.attrs.get("axis", 1)

    layer = myf('Softmax', node_name, input_name_list, [output_name], axis=axis)

    return layer

def _convert_pool(node, graph, err):
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    if node.op_type.endswith("MaxPool"):
        pool_type = P.Pooling.MAX
    elif node.op_type.endswith("AveragePool"):
        pool_type = P.Pooling.AVE
    else:
        return err.unsupported_op_configuration(node, "Unsupported pool type")

    kernel_shape = node.attrs["kernel_shape"]
    strides = node.attrs.get('strides', [1, 1])
    pads = node.attrs.get('pads', [0, 0, 0, 0])

    layer = myf("Pooling", node_name, [input_name], [output_name], pooling_param=dict(pool=pool_type,
                                                                                      kernel_h=kernel_shape[0],
                                                                                      kernel_w=kernel_shape[1],
                                                                                      stride_h=strides[0],
                                                                                      stride_w=strides[1],
                                                                                      pad_h=pads[0],
                                                                                      pad_w=pads[1]))
    graph.channel_dims[output_name] = graph.channel_dims[input_name]
    return layer


def _convert_dropout(node, graph, err):
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    ratio = node.attrs.get('ratio', 0.5)
    layer = myf("Dropout", node_name, [input_name], [output_name], dropout_ratio=ratio)
    graph.channel_dims[output_name] = graph.channel_dims[input_name]
    return layer


def _convert_gemm(node, graph, err):
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    weight_name = node.inputs[1]
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name, ))
        return

    #if node.attrs["broadcast"] != 1 or node.attrs["transB"] != 1:
    if node.attrs["transB"] != 1:
        return err.unsupported_op_configuration(node, "Gemm is supported only for inner_product layer")

    b = None
    bias_flag = False
    if len(node.inputs) > 2:
        b = node.input_tensors[node.inputs[2]]

    if len(W.shape) != 2 or (b is not None and len(b.shape) != 1):
        return err.unsupported_op_configuration(node, "Gemm is supported only for inner_product layer")
    if b is not None:
        bias_flag = True
        if W.shape[0] != b.shape[0]:
            return err.unsupported_op_configuration(node,
                                                    "Gemm is supported only for inner_product layer")

    layer = myf("InnerProduct", node_name, [input_name], [output_name], num_output=W.shape[0], bias_term=bias_flag)
    graph.channel_dims[output_name] = W.shape[0]

    return layer


def _convert_matmul(node, graph, err):  # 建立网络结构图

    node_name = node.name
    input_name = str(node.inputs[0])  # 上层节点名
    output_name = str(node.outputs[0])  # 输出节点名
    weight_name = node.inputs[1]  # 本层参数名
    if weight_name in node.input_tensors:  # 判断参数数组是否真的存在
        W = node.input_tensors[weight_name]  # 获得参数数组
    else:  # 没有的话也就没意义继续了
        err.missing_initializer(node,
                                "MatMul weight tensor: {} not found in the graph initializer".format(weight_name, ))
        return

    b = None
    bias_flag = False
    if len(node.inputs) > 2:  # 如果只有上层节点名和 W 权值，则为 2
        b = node.input_tensors[node.inputs[2]]
    # 权值 shape 不对，也没意义继续了
    if len(W.shape) != 2 or (b is not None and len(b.shape) != 1):
        return err.unsupported_op_configuration(node, "MatMul is supported only for inner_product layer")
    if b is not None:
        bias_flag = True
        if W.shape[1] != b.shape[0]:  # FC 中，二者 shape[0] 是输出通道数， 一定相等，shape[1] 是输入通道数。
            return err.unsupported_op_configuration(node,
                                                    "MatMul is supported only for inner_product layer")
    # 不同于 gemm ，matmul 不做转置操作，w = (A, B), A 是输入通道数， B 是输出通道数
    layer = myf("InnerProduct", node_name, [input_name], [output_name], num_output=W.shape[1], bias_term=bias_flag)
    graph.channel_dims[output_name] = W.shape[1]  # 获得输出通道数

    return layer


def _convert_upsample(node,graph,err):
    # factor_list = node.input_tensors.get(node.inputs[1])
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    mode = node.attrs["mode"]
    print(mode)
    #https://github.com/pytorch/pytorch/issues/6900
    if  str(mode,encoding="gbk") == "linear": #mode=="linear":
        # factor = int(node.attrs["scales"])
        # input_shape = graph.shape_dict[input_name]
        # channels = input_shape[1]
        channels = graph.channel_dims[input_name]
        # pad = int(math.ceil((factor - 1) / 2.))
        # layer = myf("Deconvolution", node_name, [input_name], [output_name],
        #             kernel_size=2 * factor - factor % 2,
        #             stride=factor, group=channels,
        #             pad = pad, num_output=channels, bias_term = False)
        scales = node.input_tensors.get(node.inputs[1])
        height_scale = int(scales[2])
        width_scale = int(scales[3])
        pad_h = int(math.ceil((height_scale - 1) / 2.))
        pad_w = int(math.ceil((width_scale - 1) / 2.))
        layer = myf("Deconvolution", node_name, [input_name], [output_name],
                    convolution_param=dict(
                        num_output=channels,
                        # kernel_size=(int(2 * height_scale - height_scale % 2),int(2 * width_scale - width_scale % 2)),
                        # stride=(height_scale,width_scale),
                        # pad=(pad_h,pad_w),
                        kernel_h=int(2 * height_scale - height_scale % 2),
                        kernel_w=int(2 * width_scale - width_scale % 2),
                        stride_h=height_scale,
                        stride_w=height_scale,
                        pad_h=pad_h,
                        pad_w=pad_w,
                        group=channels,
                        bias_term=False,
                        weight_filler=dict(type="bilinear")
                    ),param=dict(lr_mult=0,decay_mult=0))
    # https://github.com/jnulzl/caffe_plus 里面的upsample 是用的nearest插值
    elif str(mode,encoding="gbk") == "nearest":
        scales = node.input_tensors.get(node.inputs[1])
        height_scale = scales[2]
        width_scale = scales[3]
        layer = myf("Upsample", node_name, [input_name], [output_name],
                    upsample_param=dict(
                        mode = int(0),
                        height_scale = int(height_scale),
                        width_scale = int(width_scale)
                    ))
    else:
        scales = node.input_tensors.get(node.inputs[1])
        height_scale = int(scales[2])
        width_scale = int(scales[3])
        # factor = int(node.attrs["scales"])
        # input_shape = graph.shape_dict[input_name]
        # channels = input_shape[1]
        channels = graph.channel_dims[input_name]
        print(height_scale)
        print(width_scale)

        layer = myf("Deconvolution", node_name, [input_name], [output_name],
                    convolution_param=dict(
                        num_output=channels,
                        # kernel_size=(height_scale,width_scale),
                        # stride=(height_scale,width_scale),
                        kernel_h=height_scale,
                        kernel_w=width_scale,
                        stride_h=height_scale,
                        stride_w=height_scale,
                        group=channels,
                        bias_term=False,
                    ))

    graph.channel_dims[output_name] = graph.channel_dims[input_name]
    return layer

def _convert_resize(node,graph,err):
    # factor_list = node.input_tensors.get(node.inputs[1])
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    mode = node.attrs["mode"]
    #https://github.com/pytorch/pytorch/issues/6900
    if  str(mode,encoding="gbk") == "linear": #mode=="linear":
        # factor = int(node.attrs["scales"])
        # input_shape = graph.shape_dict[input_name]
        # channels = input_shape[1]
        channels = graph.channel_dims[input_name]
        # pad = int(math.ceil((factor - 1) / 2.))
        # layer = myf("Deconvolution", node_name, [input_name], [output_name],
        #             kernel_size=2 * factor - factor % 2,
        #             stride=factor, group=channels,
        #             pad = pad, num_output=channels, bias_term = False)
        scales = node.input_tensors.get(node.inputs[1])
        height_scale = int(scales[2])
        width_scale = int(scales[3])
        pad_h = int(math.ceil((height_scale - 1) / 2.))
        pad_w = int(math.ceil((width_scale - 1) / 2.))
        layer = myf("Deconvolution", node_name, [input_name], [output_name],
                    convolution_param=dict(
                        num_output=channels,
                        # kernel_size=(int(2 * height_scale - height_scale % 2),int(2 * width_scale - width_scale % 2)),
                        # stride=(height_scale,width_scale),
                        # pad=(pad_h,pad_w),
                        kernel_h=int(2 * height_scale - height_scale % 2),
                        kernel_w=int(2 * width_scale - width_scale % 2),
                        stride_h=height_scale,
                        stride_w=height_scale,
                        pad_h=pad_h,
                        pad_w=pad_w,
                        group=channels,
                        bias_term=False,
                        weight_filler=dict(type="bilinear")
                    ),param=dict(lr_mult=0,decay_mult=0))
    # https://github.com/jnulzl/caffe_plus 里面的upsample 是用的nearest插值
    elif str(mode,encoding="gbk") == "nearest":
        scales = node.input_tensors.get(node.inputs[1])
        height_scale = scales[2]
        width_scale = scales[3]
        layer = myf("Upsample", node_name, [input_name], [output_name],
                    upsample_param=dict(
                        mode = int(0),
                        height_scale = int(height_scale),
                        width_scale = int(width_scale)
                    ))
    else:
        scales = node.input_tensors.get(node.inputs[1])
        height_scale = int(scales[2])
        width_scale = int(scales[3])
        # factor = int(node.attrs["scales"])
        # input_shape = graph.shape_dict[input_name]
        # channels = input_shape[1]
        channels = graph.channel_dims[input_name]
        print(height_scale)
        print(width_scale)

        layer = myf("Deconvolution", node_name, [input_name], [output_name],
                    convolution_param=dict(
                        num_output=channels,
                        # kernel_size=(height_scale,width_scale),
                        # stride=(height_scale,width_scale),
                        kernel_h=height_scale,
                        kernel_w=width_scale,
                        stride_h=height_scale,
                        stride_w=height_scale,
                        group=channels,
                        bias_term=False,
                    ))

    graph.channel_dims[output_name] = graph.channel_dims[input_name]
    return layer

def _convert_resize_(node,graph,err):
    # factor_list = node.input_tensors.get(node.inputs[1])
    node_name = node.name
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    mode = node.attrs["mode"]
    if str(mode,encoding="gbk") == "linear":
        scales = node.input_tensors.get(node.inputs[1])
        layer = myf("Upsample", node_name, [input_name], [output_name],
                    upsample_param=dict(
                        mode = int(1),
                        height_scale = 1,
                        width_scale = 1
                    ))
    graph.channel_dims[output_name] = graph.channel_dims[input_name]
    return layer


def _convert_concat(node, graph, err):
    node_name = node.name
    input_name_list = [str(i) for i in node.inputs]
    output_name = str(node.outputs[0])
    axis = node.attrs.get("axis", 1)

    layer = myf('Concat', node_name, input_name_list, [output_name], axis=axis)
    if axis == 1:
        dim = 0
        for name in input_name_list:
            dim += graph.channel_dims[name]
        graph.channel_dims[output_name] = dim
    else:
        graph.channel_dims[output_name] = graph.channel_dims[input_name_list[0]]

    return layer


def _convert_conv_transpose(node, graph, err):  # 反卷积
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    node_name = node.name
    weight_name = node.inputs[1]
    W = None
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name, ))
    bias_flag = False
    bias = None
    if len(node.inputs) > 2:
        bias = node.input_tensors[node.inputs[2]]
        bias_flag = True
    dilations = node.attrs.get("dilations", [1, 1])
    # groups = 1
    groups = node.attrs.get("group", 1)
    kernel_shape = node.attrs["kernel_shape"]
    pads = node.attrs.get("pads", [0, 0, 0, 0])
    strides = node.attrs["strides"]

    layer = myf('Deconvolution', node_name, [input_name], [output_name],
                convolution_param=dict(
                    num_output=W.shape[0],  # 此处我做了更改，分组卷积需要注意
                    kernel_h=kernel_shape[0], kernel_w=kernel_shape[1],
                    stride_h=strides[0], stride_w=strides[1],
                    group=groups,
                    pad_h=pads[0], pad_w=pads[1],
                    bias_term=bias_flag,
                ))

    graph.channel_dims[output_name] = W.shape[1]
    return layer


def _convert_PassThrough(node_name, input_name, output_name, input_channel, block_height, block_width):  # 反卷积

    layer = myf('PassThrough', node_name, [input_name], [output_name],
                pass_through_param=dict(
                    num_output=input_channel * block_height * block_width,
                    block_height=block_height,
                    block_width=block_width,
    ))

    return layer

def _convert_Reorg(graph, node_name, input_name, output_name):

    layer = myf('Reorg', node_name, [input_name], [output_name],
                reorg_param=dict(
                    stride=2,
                    reverse = False,
    ))
    graph.channel_dims[output_name] = graph.channel_dims[input_name]
    return layer

def _convert_conv_slice(node, graph, err):
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    node_name = node.name
    print("***********_convert_conv_slice***********")
    print("input_name: ", node.inputs[0])
    print("output_name: ", node.outputs[0])
    print("node_name: ", node.name)

    input_tensors = node.input_tensors
    input_tensor_keys = list(node.input_tensors.keys())
    print("input_tensor_keys: ", input_tensor_keys)

    axes = input_tensors[input_tensor_keys[2]]
    print("axes: ", axes)
    #graph.channel_dims[input_name] = graph.shape_dict[input_name][1]
    #channels = graph.shape_dict[input_name][1]
    #channels = graph.channel_dims[input_name]
    channels = graph.shape_dict[input_name][1]
    print("channels: ", channels)

    if len(axes) != 1:
        return err.unsupported_op_configuration(node, "Only single axis Slice is supported now")
    starts = input_tensors[input_tensor_keys[0]]
    print("starts: ", starts)
    ends = input_tensors[input_tensor_keys[1]]
    print("ends: ", ends)
    start = starts[0]
    end = ends[0]
    valid_pts = []
    for pt in [0, 320]:
        if pt is not None and pt != 0 and pt != channels:
            valid_pts.append(pt)
    if start == 0:
        output_name_list = [output_name, str(output_name) + "slice_another"]
    else:
        output_name_list = [str(output_name) + "slice_another", output_name]

    if len(axes) == 0: axes = range(len(starts))
    if len(axes) == 1:
        if axes[0] == 1:
            axis = 'channel'
        elif axes[0] == 2:
            axis = 'height'
        elif axes[0] == 3:
            axis = 'width'
        else:
            return err.unsupported_op_configuration(node, "Slice is supported only along H, W or C dimensions")
    else:
        return err.unsupported_op_configuration(node,"Slice is supported only along one axis for 3D or 4D Tensors")
    layer = myf('Slice', node_name, [input_name], output_name_list, slice_dim=axes[0], slice_point=320)
    graph.channel_dims[output_name_list[0]] = graph.channel_dims[input_name]
    graph.channel_dims[output_name_list[-1]] = graph.channel_dims[input_name]
    return layer

def _convert_conv_split(node, graph, err):
    input_name = str(node.inputs[0])
    output_name1 = str(node.outputs[0])
    output_name2 = str(node.outputs[1])
    print([output_name1,output_name2])
    node_name = node.name
    axis = node.attrs.get("axis", 1)
    split = node.attrs.get("split", [0,0])[0]

    layer = myf('Slice', node_name, [input_name], [output_name1,output_name2], slice_dim=axis, slice_point=[0,split])
    graph.channel_dims[output_name1] = graph.channel_dims[input_name]
    graph.channel_dims[output_name2] = graph.channel_dims[input_name]
    return layer
      

_ONNX_NODE_REGISTRY = {
    "Conv": _convert_conv,
    "Relu": _convert_relu,
    "LeakyRelu": _convert_leaky_relu,
    "PRelu": _convert_prelu,
    "BatchNormalization": _convert_BatchNorm,
    "Add": _convert_Add,
    "Mul": _convert_Mul,
    "Reshape": _convert_Reshape,
    "MaxPool": _convert_pool,
    "AveragePool": _convert_pool,
    "Dropout": _convert_dropout,
    "Gemm": _convert_gemm,
    "MatMul": _convert_matmul,
    "Upsample": _convert_upsample,
    "Resize":_convert_resize,
    "Concat": _convert_concat,
    "ConvTranspose": _convert_conv_transpose,
    "Sigmoid": _convert_sigmoid,
    "Slice": _convert_conv_slice,
    "Split": _convert_conv_split,
    "Flatten": _convert_Flatten,
    "Transpose": _convert_Permute,
    "Softmax": _convert_Softmax,
    "PassThrough": _convert_PassThrough,
    "Reorg":_convert_Reorg
}
