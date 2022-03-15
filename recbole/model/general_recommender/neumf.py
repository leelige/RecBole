# -*- coding: utf-8 -*-
# @Time   : 2020/6/27
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/22,
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmain.com

r"""
NeuMF
################################################
Reference:
    Xiangnan He et al. "Neural Collaborative Filtering." in WWW 2017.
"""

import torch
import torch.nn as nn
from torch.nn.init import normal_

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.layers import MLPLayers
# MLPLayers(MLP中每一层的size(以list的形式输入),dropout的概率(default=0)，激活函数(default=relu))
from recbole.utils import InputType


class NeuMF(GeneralRecommender):
    r"""NeuMF is an neural network enhanced matrix factorization model.
    It replace the dot product to mlp for a more precise user-item interaction.

    Note:

        Our implementation only contains a rough pretraining function.

        GMF和MLP采用超参数α去trade-off
        论文里使用的是adam方法替代SGD

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NeuMF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        # load parameters info
        self.mf_embedding_size = config['mf_embedding_size']    # mf的embedding,default=64
        self.mlp_embedding_size = config['mlp_embedding_size']  # mlp的embedding,default=64
        self.mlp_hidden_size = config['mlp_hidden_size']        # mlp隐藏层的大小，default=[128,64]
        self.dropout_prob = config['dropout_prob']              # dropout的概率,default=0.1
        self.mf_train = config['mf_train']                      # bool值,是否训练MF,default=True
        self.mlp_train = config['mlp_train']                    # bool值，是否训练MLP,default=True
        self.use_pretrain = config['use_pretrain']              # bool值,default=False
        self.mf_pretrain_path = config['mf_pretrain_path']
        self.mlp_pretrain_path = config['mlp_pretrain_path']

        # define layers and loss
        self.user_mf_embedding = nn.Embedding(self.n_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(self.n_items, self.mf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(self.n_users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(self.n_items, self.mlp_embedding_size)
        self.mlp_layers = MLPLayers([2 * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout_prob)
        # 等价于self.mlp_layers = MLPLayers([128,128,64],0.1)
        self.mlp_layers.logger = None  # remove logger to use torch.save()
        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size + self.mlp_hidden_size[-1], 1)
            # nn.Linear(input_size,output_size),这里将mf的embedding和mlp的最后一个隐藏层的神经元个数相加，对应原论文的拼接思想，最后输出的是一个值
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, 1)
        elif self.mlp_train:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid() # 对user和item的latent vector进行
        self.loss = nn.BCELoss()  # 二分类交叉熵，在此Loss层需先sigmoid

        # parameters initialization
        if self.use_pretrain:
            self.load_pretrain()
        else:
            self.apply(self._init_weights)

    def load_pretrain(self):
        r"""A simple implementation of loading pretrained parameters.

        """
        mf = torch.load(self.mf_pretrain_path)
        mlp = torch.load(self.mlp_pretrain_path)
        self.user_mf_embedding.weight.data.copy_(mf.user_mf_embedding.weight)    # 原论文使用adam训练模型，这里将训练好的参数存入
        self.item_mf_embedding.weight.data.copy_(mf.item_mf_embedding.weight)
        self.user_mlp_embedding.weight.data.copy_(mlp.user_mlp_embedding.weight)
        self.item_mlp_embedding.weight.data.copy_(mlp.item_mlp_embedding.weight)

        for (m1, m2) in zip(self.mlp_layers.mlp_layers, mlp.mlp_layers.mlp_layers):
            if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                m1.weight.data.copy_(m2.weight)
                m1.bias.data.copy_(m2.bias)
                # 将预训练好的参数替换原始的参数

        predict_weight = torch.cat([mf.predict_layer.weight, mlp.predict_layer.weight], dim=1)
        # 评估层需要进行concat，dim=1表示横着拼，这两个是矩阵
        predict_bias = mf.predict_layer.bias + mlp.predict_layer.bias

        self.predict_layer.weight.data.copy_(0.5 * predict_weight)  # 设置权值进行trade-off
        self.predict_layer.weight.data.copy_(0.5 * predict_bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)  # 随机标准化参数

    def forward(self, user, item):                  # 为predict()和caculate_loss()服务
                                                    # interaction()生成的是一个tensor
                                                    # user=interaction(user_id) 即 interaction([a,a,b,c,c])
        user_mf_e = self.user_mf_embedding(user)    # user是user_id,输出的是每一个id的tensor,输入的时候是以batch_size的形式
        item_mf_e = self.item_mf_embedding(item)    # item是item_id,输出的是每一个item的tensor,输入的时候是以batch_size的形式
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        if self.mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
            # 举个例子：比如有3个user（a，b，c），3个item（1，2，3） a和1，2交互；b和3有交互；c和1，3有交互，
            # 那么就有(a,1),(a,2),(b,3),(c,1),(c,3),这里不讨论batch
            # user_mf_e=(a,a,b,c,c), item_mf_e=(1,2,3,1,3)
        if self.mlp_train:
            mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e), -1))  # [batch_size, layers[-1]]
            # torch里的-1表示横着拼接
            # 输出的是一个经过MLP后的矩阵，行向量表示每一个拼接的user和item ，比如：(a,1) ,(a,2),(a,3)
        if self.mf_train and self.mlp_train:  # 再将拼接后的进行判断
            output = self.sigmoid(self.predict_layer(torch.cat((mf_output, mlp_output), -1)))
        elif self.mf_train:
            output = self.sigmoid(self.predict_layer(mf_output))
        elif self.mlp_train:
            output = self.sigmoid(self.predict_layer(mlp_output))
        else:
            raise RuntimeError('mf_train and mlp_train can not be False at the same time')
        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output = self.forward(user, item)
        return self.loss(output, label)  # 交叉熵 输出值（output）-目标值（label）

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def dump_parameters(self):  # 保存模型参数
        r"""A simple implementation of dumping model parameters for pretrain.

        """
        if self.mf_train and not self.mlp_train:
            save_path = self.mf_pretrain_path
            torch.save(self, save_path)
        elif self.mlp_train and not self.mf_train:
            save_path = self.mlp_pretrain_path
            torch.save(self, save_path)

