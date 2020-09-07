# -*- coding: utf-8 -*-
"""
train
"""
from __future__ import print_function
import os
import numpy as np
import argparse

import paddle
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
from paddle.fluid.incubate.fleet.base import role_maker


#output_dir = "/root/paddlejob/workspace/output/"
#print(output_dir)

parser = argparse.ArgumentParser(description='Argument settings')
# parser.add_argument('--distributed', action='store_true', default=False,
#     help='distributed training flag for showing the differences in code that'
#          ' distinguish distributed training from single-card training. NOTICE:'
#          ' Do not specify this flag in single-card training mode')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epoch_num', type=int, default=1,
    help='epoch number for training')
parser.add_argument('--rdlearning_rate', type=float, default=4e-3,
    help='rdlearning rate')
parser.add_argument('--fdlearning_rate', type=float, default=4e-3,
    help='fdlearning rate')
parser.add_argument('--glearning_rate', type=float, default=1e-3,
    help='glearning rate')
parser.add_argument('--dataset_base_path', type=str,
    default="/root/paddlejob/workspace/train_data/datasets/data65/",
    help='dataset path')
parser.add_argument('--output_base_path', type=str, default="/root/paddlejob/workspace/output/",
    help='output path')

parser.add_argument('--load_model', type=bool, default=False,
    help='if load_model or not')
parser.add_argument('--draw', type=bool, default=False,
    help='if show imgs')
parser.add_argument('--saveimg', type=bool, default=False,
    help='if save imgs to file')
parser.add_argument('--model_path', type=str, default="./output",
    help='where to load or save model')

#load_model=False, draw=False, model_path = './output/', n_class=10, saveimg=False
args = parser.parse_args()

print(args)

#定义
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, Linear
import numpy as np
import matplotlib.pyplot as plt
import time

# 噪声维度
Z_DIM = 128
BATCH_SIZE = 128
n_class = 10
# 读取真实图片的数据集，这里去除了数据集中的label数据，因为label在这里使用不上，这里不考虑标签分类问题。
def mnist_reader(reader):
    def r():
        for img, label in reader():
            yield img.reshape(1, 28, 28)
    return r
def cifar10_reader(reader):
    def r():
        for img, label in reader():
            yield img.reshape(3, 32, 32)
    return r

# 噪声生成，通过由噪声来生成假的图片数据输入。
def z_reader():
    while True:
        yield np.random.normal(0.0, 1.0, (Z_DIM, 1, 1)).astype('float32')                #正态分布，正态分布的均值、标准差、参数

# 生成真实图片reader
# mnist_generator = paddle.batch(
#         paddle.reader.shuffle(mnist_reader(paddle.dataset.mnist.train()), 30000),
#         batch_size=BATCH_SIZE)
cifar10_generator = paddle.batch(
    paddle.reader.shuffle(cifar10_reader(paddle.dataset.cifar.train10()), 30000),
    batch_size = BATCH_SIZE)

# 生成假图片的reader
z_generator = paddle.batch(z_reader, batch_size=BATCH_SIZE)

#定义网络以及G和D
import numpy as np
from scipy.stats import truncnorm
import paddle
import paddle.fluid as fluid
from paddle.fluid import layers, dygraph as dg
from paddle.fluid.initializer import Normal, Constant, Uniform


def l2normalize(v, eps=1e-4):
  return layers.l2_normalize(v, -1, epsilon=eps)
 
class ReLU(dg.Layer):
  def forward(self, x):
    return layers.relu(x)
    
 
class SoftMax(dg.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.kwargs = kwargs
  
  def forward(self, x):
    return layers.softmax(x, **self.kwargs)
 
 
# 此批归一化增加了累计平均和累计方差功能，在完成训练后，验证过程前，
# 开启累计过程，在多个step下估计更准确的平均和方差
class BatchNorm(dg.BatchNorm):
  def __init__(self, *args, **kwargs):
    if 'affine' in kwargs:
      affine = kwargs.pop('affine')
      if not affine:
        kwargs['param_attr'] = fluid.ParamAttr(initializer=fluid.initializer.Constant(value=1.0), trainable=False)
        kwargs['bias_attr'] = fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0), trainable=False)
    else:
      affine = True
    
    super().__init__(*args, **kwargs)
    self.affine = affine
    self.initialized = False
    self.accumulating = False
    self.accumulated_mean = self.create_parameter(shape=[args[0]], default_initializer=Constant(0.0))
    self.accumulated_var = self.create_parameter(shape=[args[0]], default_initializer=Constant(0.0))
    self.accumulated_counter = self.create_parameter(shape=[1], default_initializer=Constant(1e-12))
    self.accumulated_mean.trainable = False
    self.accumulated_var.trainable = False
    self.accumulated_counter.trainable = False

  def forward(self, inputs, *args, **kwargs):
    if not self.initialized:
      self.check_accumulation()
      self.set_initialized(True)
    if self.accumulating:
      self.eval()
      with dg.no_grad():
        axes = [0] + ([] if len(inputs.shape) == 2 else list(range(2,len(inputs.shape))))
        _mean = layers.reduce_mean(inputs, axes, keep_dim=True)
        mean = layers.reduce_mean(inputs, axes, keep_dim=False)
        var = layers.reduce_mean((inputs-_mean)**2, axes)
        self.accumulated_mean.set_value((self.accumulated_mean*self.accumulated_counter + mean) / (self.accumulated_counter + 1))
        self.accumulated_var.set_value((self.accumulated_var*self.accumulated_counter + var) / (self.accumulated_counter + 1))
        self.accumulated_counter.set_value(self.accumulated_counter + 1)
        _mean = self._mean*1.0
        _variance = self.variance*1.0
        self._mean.set_value(self.accumulated_mean)
        self._variance.set_value(self.accumulated_var)
        out = super().forward(inputs, *args, **kwargs)
        self._mean.set_value(_mean)
        self._variance.set_value(_variance)
        return out
    out = super().forward(inputs, *args, **kwargs)
    return out

  def check_accumulation(self):
    if self.accumulated_counter.numpy().mean() > 1-1e-12:
      self._mean.set_value(self.accumulated_mean)
      self._variance.set_value(self.accumulated_var)
      return True
    return False

  def clear_accumulated(self):
    self.accumulated_mean.set_value(self.accumulated_mean*0.0)
    self.accumulated_var.set_value(self.accumulated_var*0.0)
    self.accumulated_counter.set_value(self.accumulated_counter*0.0+1e-2)

  def set_accumulating(self, status=True):
    if status == True:
      self.accumulating = True
    else:
      self.accumulating = False

  def set_initialized(self, status=False):
    if status == False:
      self.initialized = False
    else:
      self.initialized = True
      
  def train(self):
    super().train()
    if self.affine:
      self.weight.stop_gradient = False
      self.bias.stop_gradient = False
    else:
      self.weight.stop_gradient = True
      self.bias.stop_gradient = True
    self._use_global_stats = False
    
  def eval(self):
    super().eval()
    self.weight.stop_gradient = True
    self.bias.stop_gradient = True
    self._use_global_stats = True
 

# 此谱归一化继承自Paddle动态图自身的谱归一化，v权重与tf的实现不同，但是v本身是根据u求的，
# 所以做权重转换时不需要载入v的权重
class SpectralNorm(dg.SpectralNorm):
  def __init__(self, module, weight_name='weight', power_iterations=1, **kwargs):
    weight_shape = getattr(module, weight_name).shape
    if 'dim' not in kwargs:
      if isinstance(module, ( # dg.Conv1D, dg.Conv1DTranspose,
                          dg.Conv2D, dg.Conv2DTranspose,
                          dg.Conv3D, dg.Conv3DTranspose)):
          kwargs['dim'] = 0
      else:
          kwargs['dim'] = 1
    kwargs['power_iters'] = power_iterations
    if 'weight_shape' in kwargs:
      kwargs.pop('weight_shape')
    super().__init__(weight_shape, **kwargs)
    self.weight = getattr(module, weight_name)

    del module._parameters[weight_name]
    self.module = module
    self.weight_name = weight_name
  
  def forward(self, *args, **kwargs):
    weight_norm = super().forward(self.weight)
    setattr(self.module, self.weight_name, weight_norm)
    out = self.module(*args, **kwargs)
    return out


# 以下这个谱归一化参考自PyTorch的实现，但是与PyTorch对比过程中，发现两次matmul后的结果精度相差极大
# class SpectralNorm(dg.Layer):
#   def __init__(self, module, name='weight', power_iterations=1):
#     super().__init__()
#     self.module = module
#     self.name = name
#     self.power_iterations = power_iterations
#     if not self._made_params():
#       self._make_params()

#   def _update_u_v(self):
#     u = getattr(self.module, self.name + "_u")
#     v = getattr(self.module, self.name + "_v")
#     w = getattr(self.module, self.name + "_bar")

#     height = w.shape[0]
#     _w = layers.reshape(w,(height, -1))
#     for _ in range(self.power_iterations):
#       v = l2normalize(layers.matmul(layers.transpose(_w,[1,0]), u))
#       u = l2normalize(layers.matmul(_w, v))

#     sigma = layers.matmul(u,layers.matmul(_w, v))
#     setattr(self.module, self.name, w / sigma)
#     getattr(self.module, self.name + "_u").set_value(u)

#   def _made_params(self):
#     try:
#       getattr(self.module, self.name + "_u")
#       getattr(self.module, self.name + "_v")
#       getattr(self.module, self.name + "_bar")
#       return True
#     except AttributeError:
#       return False

#   def _make_params(self):
#     w = getattr(self.module, self.name)

#     height = w.shape[0]
#     width = layers.reshape(w,(height, -1)).shape[1]

#     u = self.create_parameter(shape=[height], default_initializer=Normal(0, 1))
#     u.stop_gradient = True
#     v = self.create_parameter(shape=[height], default_initializer=Normal(0, 1))
#     u.stop_gradient = True
#     u.set_value(l2normalize(u))
#     v.set_value(l2normalize(v))
#     w_bar = w

#     del self.module._parameters[self.name]
#     self.module.add_parameter(self.name + "_bar", w_bar)
#     self.module.add_parameter(self.name + "_u", u)
#     self.module.add_parameter(self.name + "_v", v)

#   def forward(self, *args):
#     self._update_u_v()
#     return self.module.forward(*args)
    
    
class SelfAttention(dg.Layer):
  def __init__(self, in_dim, activation=layers.relu):
    super().__init__()
    self.chanel_in = in_dim
    self.activation = activation
 
    self.theta = SpectralNorm(dg.Conv2D(in_dim, in_dim // 8, 1, bias_attr=False))
    self.phi = SpectralNorm(dg.Conv2D(in_dim, in_dim // 8, 1, bias_attr=False))
    self.pool = dg.Pool2D(2, 'max', 2)
    self.g = SpectralNorm(dg.Conv2D(in_dim, in_dim // 2, 1, bias_attr=False))
    self.o_conv = SpectralNorm(dg.Conv2D(in_dim // 2, in_dim, 1, bias_attr=False))
    self.gamma = self.create_parameter([1,], default_initializer=Constant(0.0))
 
    self.softmax = SoftMax(axis=-1)
 
  def forward(self, x):
    m_batchsize, C, width, height = x.shape
    N = height * width
 
    theta = self.theta(x)
    phi = self.phi(x)
    phi = self.pool(phi)
    phi = layers.reshape(phi,(m_batchsize, -1, N // 4))
    theta = layers.reshape(theta,(m_batchsize, -1, N))
    theta = layers.transpose(theta,(0, 2, 1))
    attention = self.softmax(layers.bmm(theta, phi))
    g = layers.reshape(self.pool(self.g(x)),(m_batchsize, -1, N // 4))
    attn_g = layers.reshape(layers.bmm(g, layers.transpose(attention,(0, 2, 1))),(m_batchsize, -1, width, height))
    out = self.o_conv(attn_g)
    return self.gamma * out + x
 
 
class ConditionalBatchNorm(dg.Layer):
  def __init__(self, num_features, num_classes, epsilon=1e-4, momentum=0.1):
    super().__init__()
    self.num_features = num_features
    self.gamma_embed = SpectralNorm(dg.Linear(num_classes, num_features, bias_attr=False))
    self.beta_embed = SpectralNorm(dg.Linear(num_classes, num_features, bias_attr=False))
    self.bn_in_cond = BatchNorm(num_features, affine=False, epsilon=epsilon, momentum=momentum)
 
  def forward(self, x, y):
    gamma = self.gamma_embed(y) + 1
    beta = self.beta_embed(y)
    out = self.bn_in_cond(x)
    out = layers.reshape(gamma, (-1, self.num_features, 1, 1)) * out + layers.reshape(beta, (-1, self.num_features, 1, 1))
    return out
 

class ResBlock(dg.Layer):
  def __init__(
    self,
    in_channel,
    out_channel,
    kernel_size=[3, 3],
    padding=1,
    stride=1,
    n_class=None,
    conditional=True,
    activation=layers.relu,
    upsample=True,
    downsample=False,
    z_dim=128,
    use_attention=False
  ):
    super().__init__()
 
    if conditional:
      self.cond_norm1 = ConditionalBatchNorm(in_channel, z_dim)
 
    self.conv0 = SpectralNorm(
      dg.Conv2D(in_channel, out_channel, kernel_size, stride, padding)
    )
 
    if conditional:
      self.cond_norm2 = ConditionalBatchNorm(out_channel, z_dim)
 
    self.conv1 = SpectralNorm(
      dg.Conv2D(out_channel, out_channel, kernel_size, stride, padding)
    )
 
    self.skip_proj = False
    if in_channel != out_channel or upsample or downsample:
      self.conv_sc = SpectralNorm(dg.Conv2D(in_channel, out_channel, 1, 1, 0))
      self.skip_proj = True
 
    if use_attention:
      self.attention = SelfAttention(out_channel)
 
    self.upsample = upsample
    self.downsample = downsample
    self.activation = activation
    self.conditional = conditional
    self.use_attention = use_attention
 
  def forward(self, input, condition=None):
    out = input
 
    if self.conditional:
      out = self.cond_norm1(out, condition)
    out = self.activation(out)
    if self.upsample:
      out = layers.interpolate(out, scale=2)
    out = self.conv0(out)
    if self.conditional:
      out = self.cond_norm2(out, condition)
    out = self.activation(out)
    out = self.conv1(out)
 
    if self.downsample:
      out = layers.pool2d(out, 2, pool_type='avg', pool_stride=2)
 
    if self.skip_proj:
      skip = input
      if self.upsample:
        skip = layers.interpolate(skip, scale=2, resample='NEAREST')
      skip = self.conv_sc(skip)
      if self.downsample:
        skip = layers.pool2d(skip, 2, pool_type='avg', pool_stride=2)
    else:
      skip = input
 
    out = out + skip
 
    if self.use_attention:
      out = self.attention(out)
 
    return out
 
 
class Generator(dg.Layer):
  def __init__(self, code_dim=128, n_class=1000, chn=96, blocks_with_attention="B4", resolution=512):
    super().__init__()
 
    def GBlock(in_channel, out_channel, n_class, z_dim, use_attention):
      return ResBlock(in_channel, out_channel, n_class=n_class, z_dim=z_dim, use_attention=use_attention)
 
    self.embed_y = dg.Linear(n_class, 128, bias_attr=False)
 
    self.chn = chn
    self.resolution = resolution 
    self.blocks_with_attention = set(blocks_with_attention.split(",")) 
    self.blocks_with_attention.discard('')
 
    gblock = []
    in_channels, out_channels = self.get_in_out_channels()
    self.num_split = len(in_channels) + 1
 
    z_dim = code_dim//self.num_split + 128
    self.noise_fc = SpectralNorm(dg.Linear(code_dim//self.num_split, 4 * 4 * in_channels[0]))
 
    self.sa_ids = [int(s.split('B')[-1]) for s in self.blocks_with_attention]
 
    for i, (nc_in, nc_out) in enumerate(zip(in_channels, out_channels)):
      gblock.append(GBlock(nc_in, nc_out, n_class=n_class, z_dim=z_dim, use_attention=(i+1) in self.sa_ids))
    self.blocks = dg.LayerList(gblock)
 
    self.output_layer = dg.Sequential(
      BatchNorm(1 * chn, epsilon=1e-4),
      ReLU(), 
      SpectralNorm(dg.Conv2D(1 * chn, 3, [3, 3], padding=1))
    )
 
  def get_in_out_channels(self):
    resolution = self.resolution
    if resolution == 512:
      channel_multipliers = [16, 16, 8, 8, 4, 2, 1, 1]
    elif resolution == 256:
      channel_multipliers = [16, 16, 8, 8, 4, 2, 1]
    elif resolution == 128:
      channel_multipliers = [16, 16, 8, 4, 2, 1]
    elif resolution == 64:
      channel_multipliers = [16, 16, 8, 4, 2]
    elif resolution == 32:
      channel_multipliers = [16, 4, 4, 1] #[4, 4, 4, 4][4] [16, 4, 4, 1]
    else:
      raise ValueError("Unsupported resolution: {}".format(resolution))
    in_channels = [self.chn * c for c in channel_multipliers[:-1]]
    out_channels = [self.chn * c for c in channel_multipliers[1:]]
    return in_channels, out_channels
 
  def forward(self, input, class_id):
    codes = layers.split(input, self.num_split, 1)
    class_emb = self.embed_y(class_id)  # 128
    out = self.noise_fc(codes[0])
    # out = layers.transpose(layers.reshape(out,(out.shape[0], 4, 4, -1)),(0, 3, 1, 2))
    out = layers.reshape(out,(out.shape[0], -1, 4, 4)) # for tf pretrained model, use transpose to weight
    for i, (code, gblock) in enumerate(zip(codes[1:], self.blocks)):
      condition = layers.concat([code, class_emb], 1) #concat ()方法用于连接两个或多个数组
      out = gblock(out, condition)
 
    out = self.output_layer(out)
    return layers.tanh(out)
 
 
class Discriminator(dg.Layer):
  def __init__(self, n_class=1000, chn=96, blocks_with_attention="B2", resolution=256): 
    super().__init__()
 
    def DBlock(in_channel, out_channel, downsample=True, use_attention=False):
      return ResBlock(in_channel, out_channel, conditional=False, upsample=False, downsample=downsample, use_attention=use_attention)
 
    self.chn = chn
    self.resolution = resolution  
    self.blocks_with_attention = set(blocks_with_attention.split(",")) 
    self.blocks_with_attention.discard('')
 
    self.pre_conv = dg.Sequential(
      SpectralNorm(dg.Conv2D(3, 1 * chn, 3, padding=1)),
      ReLU(),
      SpectralNorm(dg.Conv2D(1 * chn, 1 * chn, 3, padding=1)),
      dg.Pool2D(2, pool_type='avg', pool_stride=2),
    )
    self.pre_skip = SpectralNorm(dg.Conv2D(3, 1 * chn, 1))
 
    dblock = []
    in_channels, out_channels = self.get_in_out_channels()
 
    self.sa_ids = [int(s.split('B')[-1]) for s in self.blocks_with_attention]
    print(self.sa_ids, len(self.sa_ids))
 
    for i, (nc_in, nc_out) in enumerate(zip(in_channels[:-1], out_channels[:-1])):
      dblock.append(DBlock(nc_in, nc_out, downsample=True, use_attention=(i+1) in self.sa_ids))
    dblock.append(DBlock(in_channels[-1], out_channels[-1], downsample=False, use_attention=len(out_channels) in self.sa_ids))
    self.blocks = dg.LayerList(dblock)
 
    for sa_id in self.sa_ids:
      setattr(self, f'attention_{sa_id}', SelfAttention(in_channels[sa_id]))
 
    self.final_fc = SpectralNorm(dg.Linear(16 * chn, 1))
 
    self.embed_y = dg.Embedding(size=[n_class, 16 * chn], is_sparse=False, param_attr=Uniform(-0.1,0.1))
    self.embed_y = SpectralNorm(self.embed_y)


  def get_in_out_channels(self):
    resolution = self.resolution
    if resolution == 512:
      channel_multipliers = [1, 1, 2, 4, 8, 8, 16, 16]
    elif resolution == 256:
      channel_multipliers = [1, 2, 4, 8, 8, 16, 16]
    elif resolution == 128:
      channel_multipliers = [1, 2, 4, 8, 16, 16]
    elif resolution == 64:
      channel_multipliers = [2, 4, 8, 16, 16]
    elif resolution == 32:
      channel_multipliers = [1, 4, 4, 16] #[2, 2, 2, 2] [1] [1, 4, 4, 16]
    else:
      raise ValueError("Unsupported resolution: {}".format(resolution))
    in_channels = [self.chn * c for c in channel_multipliers[:-1]]
    out_channels = [self.chn * c for c in channel_multipliers[1:]]
    return in_channels, out_channels
 
  def forward(self, input, class_id):
 
    out = self.pre_conv(input)
    out += self.pre_skip(layers.pool2d(input, 2, pool_type='avg', pool_stride=2))
    for i, dblock in enumerate(self.blocks):
      out = dblock(out)
    out = layers.relu(out)
    out = layers.reshape(out,(out.shape[0], out.shape[1], -1))
    out = layers.reduce_sum(out, 2)
    out_linear = layers.squeeze(self.final_fc(out), [1])
    class_emb = self.embed_y(class_id)
    #reduce_sum 对指定维度上的Tensor元素进行求和运算，并输出相应的计算结果。
    prod = layers.reduce_sum((out * class_emb), 1)
 
    return out_linear + prod

# 动态图版本的EMA，使得模型权重更加平滑（本身不参与训练，不修改模型权重，平滑后的权重存储于EMA自身）
# 训练中断时应另外定义一个方法存储EMA中的权重
class EMA:
  def __init__(self, model, decay=0.999):
    self.model = model
    self.decay = decay
    self.shadow = {}
    self.backup = {}

  def register(self):
    for (name, param) in self.model.named_parameters():
      if not param.stop_gradient:
        self.shadow[name] = (param * 1).detach()

  def update(self):
    for (name, param) in self.model.named_parameters():
      if not param.stop_gradient:
        assert name in self.shadow
        new_average = (1.0 - self.decay) * param + self.decay * self.shadow[name]
        self.shadow[name] = (new_average * 1).detach()

  def apply_shadow(self):
    for (name, param) in self.model.named_parameters():
      if not param.stop_gradient:
        assert name in self.shadow
        self.backup[name] = param
        param.set_value(self.shadow[name])

  def restore(self):
    for (name, param) in self.model.named_parameters():
      if not param.stop_gradient:
        assert name in self.backup
        param.set_value(self.backup[name])
    self.backup = {}

def train(cifar10_generator, epoch_num=1, batch_size=128, use_gpu=True, load_model=False, draw=False, model_path = './output/', n_class=10, saveimg=False):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    #place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
    with fluid.dygraph.guard(place):
        # 模型存储路径
        # model_path = './outputb/'
        
        #strategy = fluid.dygraph.parallel.prepare_context()
        #d = D('D')
        d = Discriminator(n_class=n_class, chn=96, blocks_with_attention="B2", resolution=32) #B2
        #d = fluid.dygraph.parallel.DataParallel(d, strategy)
        d.train()
        #g = G('G')
        g = Generator(code_dim=128, n_class=n_class, chn=96, blocks_with_attention="B2", resolution=32) #b4
        #g = fluid.dygraph.parallel.DataParallel(g, strategy)
        g.train()
        # 创建优化方法
        real_d_optimizer = fluid.optimizer.AdamOptimizer(learning_rate=4e-4, parameter_list=d.parameters())
        fake_d_optimizer = fluid.optimizer.AdamOptimizer(learning_rate=4e-4, parameter_list=d.parameters())
        g_optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-4, parameter_list=g.parameters())
        #学习率参数默认值：2e-4 全部调成1e-4试试 生成器小一点，辨别器大一点 

        # 读取上次保存的模型
        if load_model == True:
            g_para, g_opt = fluid.load_dygraph(model_path+'g')
            d_para, d_r_opt = fluid.load_dygraph(model_path+'d_o_r')
            # 上面判别器的参数已经读取到d_para了,此处无需再次读取
            _, d_f_opt = fluid.load_dygraph(model_path+'d_o_f')
            g.load_dict(g_para)
            g_optimizer.set_dict(g_opt)
            d.load_dict(d_para)
            #real_d_optimizer.set_dict(d_r_opt)
            fake_d_optimizer.set_dict(d_f_opt)
        dctime = time.time()
        iteration_num = 0
        for epoch in range(epoch_num):
            
            for i, real_image in enumerate(cifar10_generator()):
                # 丢弃不满整个batch_size的数据
                if(len(real_image) != BATCH_SIZE):
                    continue               
                iteration_num += 1                
                '''
                判别器d通过最小化输入真实图片时判别器d的输出与真值标签ones的交叉熵损失，来优化判别器的参数，
                以增加判别器d识别真实图片real_image为真值标签ones的概率。
                '''
                # 将MNIST数据集里的图片读入real_image，将真值标签ones用数字1初始化
                real_image = fluid.dygraph.to_variable(np.array(real_image))
                # [Hint: Expected input_data_type == filter_data_type, but received input_data_type:2 != filter_data_type:5.] at (/paddle/paddle/fluid/operators/conv_op.cc:173)
#float64 6 int32 2 int16 1 
                ones = fluid.dygraph.to_variable(np.ones([len(real_image)]).astype('float32')) #astype('float32')) shape[128, 1]
                #print("real_image.shape", real_image.shape,"ones.shape", ones.shape, "real_image长度", len(real_image))

                y = layers.randint(0,n_class,shape=[128]) #shape=[2]
                y_hot = layers.one_hot(layers.unsqueeze(y,[1]), depth=n_class)
                # 计算判别器d判断真实图片的概率
                p_real = d(real_image, y)
                #print(f"shape infor of p_real{p_real.shape} real_image {real_image.shape} y {y.shape}")
                #InvalidArgumentError: Broadcast dimension mismatch. Operands could not be broadcast together with the shape of X = [128, 1536] and the shape of Y = [2, 1536]. Received [128] in X is not equal to [2] in Y at i:0.
                #[Hint: Expected x_dims_array[i] == y_dims_array[i] || x_dims_array[i] <= 1 || y_dims_array[i] <= 1 == true, but received x_dims_array[i] == y_dims_array[i] || x_dims_array[i] <= 1 || y_dims_array[i] <= 1:0 != true:1.] at (/paddle/paddle/fluid/operators/elementwise/elementwise_op_function.h:160)
                # 计算判别真图片为真的损失

                #print(f"shape of p_real{p_real.shape} ones {ones.shape} y{y.shape}")
                real_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p_real, ones)
# InvalidArgumentError: Input(X) and Input(Label) shall have the same rank.But received: the rank of Input(X) is [1], the rank of Input(Label) is [2].
#   [Hint: Expected rank == labels_dims.size(), but received rank:1 != labels_dims.size():2.] at (/paddle/paddle/fluid/operators/sigmoid_cross_entropy_with_logits_op.cc:47)


                real_avg_cost = fluid.layers.mean(real_cost)
                # 反向传播更新判别器d的参数 这里注释掉，回头跟fakecost一起反向传播
                # real_avg_cost.backward()
                # real_d_optimizer.minimize(real_avg_cost)
                # d.clear_gradients()
                
                '''
                判别器d通过最小化输入生成器g生成的假图片g(z)时判别器的输出与假值标签zeros的交叉熵损失，
                来优化判别器d的参数，以增加判别器d识别生成器g生成的假图片g(z)为假值标签zeros的概率。
                '''
                # 创建高斯分布的噪声z，将假值标签zeros初始化为0
                z = next(z_generator())
                #print(f"z长度{len(z)}")
                z = fluid.dygraph.to_variable(np.array(z))
                z = paddle.fluid.layers.reshape (z , [128, -1])
                zeros = fluid.dygraph.to_variable(np.zeros([len(real_image), 1]).astype('float32')) #.astype('float32') double
                zeros = paddle.fluid.layers.reshape (zeros , [128]) #(zeros , [128, -1])
                # 判别器d判断生成器g生成的假图片的概率
                #print(f"shape infor of z {z.shape} y_hot.shape{y_hot.shape} y.shape{y.shape}") #shape of z [128, 100] y_hot.shape[128, 1000] y.shape[128]
                #print(f"shape infor of y {y.shape} y_hot{y_hot.shape}, g(z,y_hot) {g(z, y_hot).shape}")
                #dct= g(z, y_hot)
                p_fake = d(g(z,y_hot), y)

# InvalidArgumentError: Input X's width should be equal to the Y's height, but received X's shape: [128, 25, 1, 1],Y's shape: [32, 24576].
#   [Hint: Expected mat_dim_x.width_ == mat_dim_y.height_, but received mat_dim_x.width_:1 != mat_dim_y.height_:32.] at (/paddle/paddle/fluid/operators/matmul_op.cc:382)


                # 计算判别生成器g生成的假图片为假的损失
                #print(p_fake, zeros)
                fake_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p_fake, zeros)
                fake_avg_cost = fluid.layers.mean(fake_cost)
                #添加real_avg_cost到fake_avg_cost，然后一起反向传播
                fake_avg_cost += real_avg_cost
                # 反向传播更新判别器d的参数
                fake_avg_cost.backward()
                fake_d_optimizer.minimize(fake_avg_cost)
                d.clear_gradients()

                '''
                生成器g通过最小化判别器d判别生成器生成的假图片g(z)为真的概率d(fake)与真值标签ones的交叉熵损失，
                来优化生成器g的参数，以增加生成器g使判别器d判别其生成的假图片g(z)为真值标签ones的概率。
                '''
                # 生成器用输入的高斯噪声z生成假图片
                fake = g(z,  y_hot)
                # 计算判别器d判断生成器g生成的假图片的概率
                p_confused = d(fake, y)
                # 使用判别器d判断生成器g生成的假图片的概率与真值ones的交叉熵计算损失
                g_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p_confused, ones)
                g_avg_cost = fluid.layers.mean(g_cost)
                # 反向传播更新生成器g的参数
                g_avg_cost.backward()
                g_optimizer.minimize(g_avg_cost)
                g.clear_gradients()
                
                # 打印输出
                if(iteration_num % 100 == 0):

                    print('epoch =', epoch, ', batch =', i, ', real_d_loss =', real_avg_cost.numpy(),
                     ', fake_d_loss =', fake_avg_cost.numpy(), 'g_loss =', g_avg_cost.numpy(), "use time=", time.time()-dctime)
                    #show_image_grid(fake.numpy(), BATCH_SIZE, epoch)
                    dctime = time.time()
                    if draw == True:
                        show_image_grid(fake.numpy(), BATCH_SIZE, epoch)
                        print(fake.numpy().shape)
                        dct = fake.numpy()[0]
                        dct = dct.transpose(1, 2, 0)
                        plt.imshow(dct)
                        # if saveimg == True:
                        #     plt.savefig("fake" + str(epoch)+ ".png")
                        plt.show()
                    if saveimg == True:
                        for i in fake.numpy().transpose(0,2,3,1):
                            plt.imshow(i)
                            plt.savefig(model_path + "fake"+str(epoch)+"_" + str(iteration_num) + ".png")

                    

            if (epoch+1) % 50 == 0 and epoch>200 : #and fluid.dygraph.parallel.Env().local_rank == 0
                dcpath = model_path + str(epoch+1)
                fluid.save_dygraph(g.state_dict(), dcpath+'g')
                fluid.save_dygraph(g_optimizer.state_dict(), dcpath+'g')
                fluid.save_dygraph(d.state_dict(), dcpath+'d_o_r')
                #fluid.save_dygraph(real_d_optimizer.state_dict(), dcpath+'d_o_r')
                fluid.save_dygraph(d.state_dict(), dcpath+'d_o_f')
                fluid.save_dygraph(fake_d_optimizer.state_dict(), dcpath+'d_o_f')

        
        # 存储模型
        #if fluid.dygraph.parallel.Env().local_rank == 0：
        fluid.save_dygraph(g.state_dict(), model_path+'g')
        fluid.save_dygraph(g_optimizer.state_dict(), model_path+'g')
        fluid.save_dygraph(d.state_dict(), model_path+'d_o_r')
        #fluid.save_dygraph(real_d_optimizer.state_dict(), model_path+'d_o_r')
        fluid.save_dygraph(d.state_dict(), model_path+'d_o_f')
        fluid.save_dygraph(fake_d_optimizer.state_dict(), model_path+'d_o_f')

#train(cifar10_generator, epoch_num=500, batch_size=BATCH_SIZE, use_gpu=True) # 10

import matplotlib.pyplot as plt

def show_image_grid(images, batch_size=128, pass_id=None):
    fig = plt.figure(figsize=(8, batch_size/32))
    fig.suptitle("Pass {}".format(pass_id))
    gs = plt.GridSpec(int(batch_size/16), 16)
    gs.update(wspace=0.05, hspace=0.05) #wspace=0.05, hspace=0.05

    for i, image in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        dct = image.transpose(1, 2, 0)
        #dct = image
        #dct = dct/255.
        Image = dct
        Image = Image/np.amax(Image)
        Image = np.clip(Image, 0, 1)
        #plt.imshow(Image)
        #plt.imshow(Image, cmap='Greys_r')
        plt.imshow(Image,cmap='Greys_r')
    #print(image.shape, type(image))    
    plt.show()



def main():
    """
    Main function for training to illustrate steps for common distributed
    training configuration
    """
    #训练 
    BATCH_SIZE = 128 
#     parser.add_argument('--load_model', type=bool, default=False,
#     help='if load_model or not')
# parser.add_argument('--draw', type=bool, default=False,
#     help='if show imgs')
# parser.add_argument('--saveimg', type=bool, default=False,
#     help='if save imgs to file')
# parser.add_argument('--model_path', type=str, default="./output",
#     help='where to load or save model')
    train(cifar10_generator, epoch_num=args.epoch_num, batch_size=BATCH_SIZE, use_gpu=True, load_model=args.load_model, model_path = args.model_path, n_class=10, draw=args.draw, saveimg=args.saveimg) # 10




if __name__ == '__main__':
    main()
