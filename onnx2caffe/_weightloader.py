from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# from caffe import params as P
import numpy as np
from ._graph import Node, Graph


def _convert_conv(net, node, graph, err, pass_through=0):
    weight_name = node.inputs[1]
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    node_name = node.name
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
    # net.params[node_name][0].data = W
    # if bias_flag:
    #     net.params[node_name][1].data = bias
    if pass_through:
        pass_through_group = W.shape[1] // 4
        w1 = W[:, 0: pass_through_group, :, :]
        w2 = W[:, pass_through_group: pass_through_group * 2, :, :]
        w3 = W[:, pass_through_group * 2:pass_through_group * 3, :, :]
        w4 = W[:, pass_through_group * 3:pass_through_group * 4, :, :]
        W = np.concatenate((w1, w3, w2, w4), 1)
        np.copyto(net.params[node_name][0].data, W, casting='same_kind')
    else:
        np.copyto(net.params[node_name][0].data, W, casting='same_kind')
    if bias_flag:
        np.copyto(net.params[node_name][1].data, bias, casting='same_kind')


def _convert_relu(net, node, graph, err):
    pass

def _convert_leaky_relu(net, node, graph, err):
    pass

def _convert_prelu(net, node, graph, err):
    #nodeName = str(node.inputs[1])
    weight = node.input_tensors[node.inputs[1]]
    #tempMul = tempMul.reshape((1, length))
    #downMul = 1 - tempMul

    # copy weight to caffe model
    shape = weight.shape
    weight = weight.reshape((shape[0]))
    np.copyto(net.params[node.name][0].data, weight, casting='same_kind')
    #np.copyto(net.params[node.name + "_scale_down"][0].data, downMul, casting='same_kind')


def _convert_sigmoid(net, node, graph, err):
    pass

def _convert_conv_slice(net, node, graph, err):
    pass

def _convert_conv_split(net, node, graph, err):
    pass

def _convert_BatchNorm(net, node, graph, err):
    scale = node.input_tensors[node.inputs[1]]
    bias = node.input_tensors[node.inputs[2]]
    mean = node.input_tensors[node.inputs[3]]
    var = node.input_tensors[node.inputs[4]]
    node_name = node.name
    np.copyto(net.params[node_name + '_bn'][0].data, mean, casting='same_kind')
    np.copyto(net.params[node_name + '_bn'][1].data, var, casting='same_kind')
    net.params[node_name + '_bn'][2].data[...] = 1.0
    np.copyto(net.params[node_name][0].data, scale, casting='same_kind')
    np.copyto(net.params[node_name][1].data, bias, casting='same_kind')
    # net.params[node_name+'_bn'][1].data = var
    # net.params[node_name][0].data = scale
    # net.params[node_name][1].data = bias


def _convert_Add(net, node, graph, err):
    pass

def _convert_Mul(net, node, graph, err):
    pass


def _convert_Reshape(net, node, graph, err):
    pass


def _convert_Flatten(net, node, graph, err):
    pass


def _convert_pool(net, node, graph, err):
    pass


def _convert_dropout(net, node, graph, err):
    pass


def _convert_Permute(net, node, graph, err):
    pass


def _convert_Softmax(net, node, graph, err):
    pass


def _convert_gemm(net, node, graph, err):
    node_name = node.name
    weight_name = node.inputs[1]
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name, ))
    #if node.attrs["broadcast"] != 1 or node.attrs["transB"] != 1:
    if node.attrs["transB"] != 1:
        return err.unsupported_op_configuration(node, "Gemm is supported only for inner_product layer")
    b = None
    if len(node.inputs) > 2:
        b = node.input_tensors[node.inputs[2]]
    if len(W.shape) != 2 or (b is not None and len(b.shape) != 1):
        return err.unsupported_op_configuration(node, "Gemm is supported only for inner_product layer")
    if b is not None:
        if W.shape[0] != b.shape[0]:
            return err.unsupported_op_configuration(node, "Gemm is supported only for inner_product layer")
    net.params[node_name][0].data[...] = W
    net.params[node_name][1].data[...] = b


def _convert_matmul(net, node, graph, err):
    node_name = node.name
    weight_name = node.inputs[1]
    if weight_name in node.input_tensors:  # 判断是否有参数，免得下面报错
        W = node.input_tensors[weight_name]  # 如果有，获得参数数组
    else:
        err.missing_initializer(node,
                                "MatMul weight tensor: {} not found in the graph initializer".format(weight_name, ))

    b = None
    if len(node.inputs) > 2:
        b = node.input_tensors[node.inputs[2]]
    if len(W.shape) != 2 or (b is not None and len(b.shape) != 1):
        return err.unsupported_op_configuration(node, "MatMul is supported only for inner_product layer")
    if b is not None:
        if W.shape[1] != b.shape[0]:
            return err.unsupported_op_configuration(node, "MatMul is supported only for inner_product layer")

    net.params[node_name][0].data[...] = W.transpose()
    #net.params[node_name][1].data[...] = b


def _convert_upsample(net, node, graph, err):
    mode = node.attrs["mode"]
    node_name = node.name
    if mode == "nearest":
        caffe_params = net.params[node_name][0].data
        weights = np.ones(caffe_params.shape).astype("float32")
        np.copyto(net.params[node_name][0].data, weights, casting='same_kind')
        # net.params[node_name][0].data[]

def _convert_resize(net, node, graph, err):
    mode = node.attrs["mode"]
    node_name = node.name
    if mode == "nearest":
        caffe_params = net.params[node_name][0].data
        weights = np.ones(caffe_params.shape).astype("float32")
        np.copyto(net.params[node_name][0].data, weights, casting='same_kind')
        # net.params[node_name][0].data[]


def _convert_concat(net, node, graph, err):
    pass


def _convert_conv_transpose(net, node, graph, err):
    weight_name = node.inputs[1]
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    node_name = node.name
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
    # net.params[node_name][0].data = W
    # if bias_flag:
    #     net.params[node_name][1].data = bias
    np.copyto(net.params[node_name][0].data, W, casting='same_kind')
    if bias_flag:
        np.copyto(net.params[node_name][1].data, bias, casting='same_kind')


def _convert_PassThrough(node, graph, err):
    pass

def _convert_Reorg(node, graph, err):
    pass

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
