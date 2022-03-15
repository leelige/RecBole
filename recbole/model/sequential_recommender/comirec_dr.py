import random

import torch
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import torch.nn.functional as F


class ComiRec_DR(SequentialRecommender):
    r"""ComiRec_DR is a model that incorporate Capsule Network for recommendation.

        Note:
            Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
        """

    def __init__(self, config, dataset):
        super(ComiRec_DR, self).__init__(config, dataset)

        # load parameters info
        # self.ITEM_SEQ = self.ITEM_ID + config['LIST_SUFFIX']
        # self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
        # self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']  # hidden_size就是embedding_size
        self.relu_layers = config['relu_layers']
        self.routing_times = config['routing_times']
        self.interest_num = config['interest_num']
        self.loss_type = config['loss_type']
        self.hard_readout = config['hard_readout']
        self.stop_grad = True
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)  # n_items就是item_id的int形式

        #  define layers and loss
        self.relu = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.ReLU()
        )
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.w = nn.Parameter(
            torch.Tensor(1, self.max_seq_length, self.interest_num * self.hidden_size, self.hidden_size))
        for weight in self.parameters():
            torch.nn.init.kaiming_normal_(weight)

    def forward(self, item_seq):
        # item_seq.shape=(batch_size, max_len)
        mask = item_seq.clone()
        mask[mask != 0] = 1  # batch_size * max_len
        item_seq_emb = self.item_embedding(item_seq)  # [batch_size, max_len, embedding_size]
        item_seq_emb = item_seq_emb * torch.reshape(mask, (-1, self.max_seq_length, 1))
        # mask_shape = batch_size * max_seq_len

        # label_eb = self.item_embedding(label_seq)

        u = torch.unsqueeze(item_seq_emb, dim=2)
        # shape=(batch_size, max_len, 1, embedding_dim)
        # 这里batch_size相当于抽取的user个数，maxlen表示规定item序列最大长度

        item_seq_emb_hat = torch.sum(self.w[:, :self.max_seq_length, :, :] * u, dim=3)
        # shape=(batch_size, max_len, hidden_size*interest_num)

        item_seq_emb_hat = torch.reshape(item_seq_emb_hat,
                                         (-1, self.max_seq_length, self.interest_num, self.hidden_size))
        item_seq_emb_hat = torch.transpose(item_seq_emb_hat, 1, 2).contiguous()
        item_seq_emb_hat = torch.reshape(item_seq_emb_hat,
                                         (-1, self.interest_num, self.max_seq_length, self.hidden_size))

        # [b, in, s, h] in:interest_num
        if self.stop_grad:  # 截断反向传播，item_emb_hat不计入梯度计算中
            item_eb_hat_iter = item_seq_emb_hat.detach()
        else:
            item_eb_hat_iter = item_seq_emb_hat
        # b的shape=[b, in, s] 将权值初始化为0（胶囊网络算法）
        capsule_weight = torch.zeros(item_seq_emb_hat.shape[0], self.interest_num, self.max_seq_length,
                                     requires_grad=True)

        for i in range(self.routing_times):  # 动态路由传递3次
            atten_mask = torch.unsqueeze(mask, 1).repeat(1, self.interest_num, 1)  # [b, in, s]
            paddings = torch.zeros_like(atten_mask, dtype=torch.float)

            capsule_softmax_weight = F.softmax(capsule_weight, dim=-1)
            capsule_softmax_weight = torch.where(torch.eq(atten_mask, 0), paddings, capsule_softmax_weight)  # mask
            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, 2)

            if i < 2:
                # s=c*u_hat , (batch_size, interest_num, 1, max_len) * (batch_size, interest_num, max_len, hidden_size)
                interest_capsule = torch.matmul(capsule_softmax_weight,
                                                item_eb_hat_iter)  # shape=(batch_size, interest_num, 1, hidden_size)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)  # shape=(batch_size, interest_num, 1, 1)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)  # shape同上
                interest_capsule = scalar_factor * interest_capsule  # squash(s)->v,shape=(batch_size, interest_num, 1, hidden_size)

                # 更新b
                delta_weight = torch.matmul(item_eb_hat_iter,  # shape=(batch_size, interest_num, max_len, hidden_size)
                                            torch.transpose(interest_capsule, 2, 3).contiguous()
                                            # shape=(batch_size, interest_num, hidden_size, 1)
                                            )  # u_hat*v, shape=(batch_size, interest_num, seq_len, 1)
                delta_weight = torch.reshape(delta_weight, (
                    -1, self.interest_num, self.max_seq_length))  # shape=(batch_size, interest_num, max_len)
                capsule_weight = capsule_weight + delta_weight  # 更新b
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_seq_emb_hat)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        user_eb = torch.reshape(interest_capsule, (-1, self.interest_num, self.hidden_size))
        # shape=(batch_size, interest_num, hidden_size)

        # readout = self.read_out(user_eb, label_seq_emb)
        # all_items = self.item_embedding.weight
        # scores = torch.matmul(readout, all_items.transpose(1, 0))  # [batch_size, n_items]

        return user_eb

    def calculate_loss(self, interaction):
        # self.ITEM_SEQ = self.ITEM_ID + config['LIST_SUFFIX']  # LIST_SUFFIX指定序列前缀
        # self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
        # self.n_items = dataset.num(self.ITEM_ID)  # 将item_id字符串形式转化为int
        item_seq = interaction[self.ITEM_SEQ]  # 这是一个tensor (值为int),其实就是item_id由字符串类型转化为int类型后的值
        # item_seq_len = interaction[self.ITEM_SEQ_LEN]  # 同上，只不过这里表示item sequence的长度所对应的tensor

        '''
        关于interaction：
            interaction是一个字典,
            举个例子：
            interaction=Interaction({'4': torch.zeros(5), '3': torch.zeros(7)})（Interaction在recbole.data.interaction中）
            输出：
            The batch_size of interaction: 7
                4, torch.Size([5]), cpu, torch.float32
                3, torch.Size([7]), cpu, torch.float32
            
            interaction['4'] ==>形式等价于上面 interaction[self.ITEM_SEQ]
            = tensor([0., 0., 0., 0., 0.])
            
            batch_size取最大值，这里最大的是7，取7【注：batch_size表示第0维】
            
        '''
        # seq_output = self.forward(item_seq, item_seq_len)
        # k = random.choice((range(4, item_seq.shape[0])
        # k = random.choice((range(4, len(list(item_seq)))))

        # k = random.choice(4, item_seq.shape[0])
        # label_seq = []  # item_id 标注的序列
        # mask = []  # 以mask的形式标注的序列
        #
        # if k >= item_seq_len:
        #     label_seq.append(list(item_seq.numpy().keys())[k - item_seq_len:k])
        #     mask.append(list(item_seq.numpy().keys())[[1.0] * item_seq_len])
        # else:
        #     label_seq.append(list(item_seq.numpy().keys())[:k] + [0] * (item_seq_len - k))
        #     mask.append(list(item_seq.numpy().keys())[[1.0] * k + [0.0] * (item_seq_len - k)])
        #
        # label_seq = list(map(float, label_seq))
        # mask = list(map(float, mask))
        # label_seq = torch.tensor(label_seq)
        # mask = torch.tensor(mask)
        seq_output = self.forward(item_seq)
        pos_items = interaction[self.POS_ITEM_ID]  # POS_ITEM_ID就是item_id

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits.sum(1), pos_items)

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]  # shape=(batch_size,max_len)
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq)
        label_eb = self.item_embedding(test_item)
        atten = torch.matmul(seq_output,  # shape=(batch_size, interest_num, hidden_size)
                             torch.reshape(label_eb, (-1, self.hidden_size, 1))
                             # shape=(batch_size, hidden_size, 1)
                             )  # shape=(batch_size, interest_num, 1)
        atten = F.softmax(torch.pow(torch.reshape(atten, (-1, self.interest_num)), 1),
                          dim=-1)  # shape=(batch_size, interest_num)

        if self.hard_readout:  # 选取interest_num个兴趣胶囊中的一个，MIND和ComiRec都是用的这种方式
            readout = torch.reshape(seq_output, (-1, self.hidden_size))[
                (torch.argmax(atten, dim=-1) + torch.arange(label_eb.shape[0]) * self.interest_num).long()]
        else:  # 综合interest_num个兴趣胶囊，论文及代码实现中没有使用这种方法
            readout = torch.matmul(torch.reshape(atten, (label_eb.shape[0], 1, self.interest_num)), seq_output)
            # shape=(batch_size, 1, interest_num)
            # shape=(batch_size, interest_num, hidden_size)
            # shape=(batch_size, 1, hidden_size)
            readout = torch.reshape(readout, (seq_output.shape[0], self.hidden_size))  # shape=(batch_size, hidden_size)
            # scores是vu堆叠成的矩阵（一个batch的vu）（vu可以说就是最终的用户嵌入）

        # scores = torch.mul(readout, label_eb).sum(dim=1)  # [B]
        scores = readout.sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):  # interaction是一个字典
        item_seq = interaction[self.ITEM_SEQ]  # shape=(batch_size,max_len)
        test_item = interaction[self.ITEM_ID]  # batch_size * n
        # item_seq_len = interaction[self.ITEM_SEQ_LEN]  # shape=(batch_size, seq_len)
        # k = random.choice((range(4, item_seq.shape[0])
        k = random.choice(range(4, item_seq.shape[0]))
        # label_seq = []
        # mask = []
        # if k >= item_seq_len:
        #     label_seq.append(list(item_seq.numpy().keys())[k - item_seq_len:k])
        #     mask.append(list(item_seq.numpy().keys())[[1.0] * item_seq_len])
        # else:
        #     label_seq.append(list(item_seq.numpy().keys())[:k] + [0] * (item_seq_len - k))
        #     mask.append(list(item_seq.numpy().keys())[[1.0] * k + [0.0] * (item_seq_len - k)])
        # label_seq = list(map(float, label_seq))
        # mask = list(map(float, mask))
        # label_seq = torch.tensor(label_seq)
        # mask = torch.tensor(mask)
        seq_output = self.forward(item_seq)
        # 这里seq_output就是user_eb
        # 这个模型训练过程中label是可见的，此处的item_eb就是label物品的嵌入
        label_eb = self.item_embedding(test_item)
        atten = torch.matmul(seq_output,  # shape=(batch_size, interest_num, hidden_size)
                             torch.reshape(label_eb, (-1, self.hidden_size, 1))
                             # shape=(batch_size, hidden_size, 1)
                             )  # shape=(batch_size, interest_num, 1)
        atten = F.softmax(torch.pow(torch.reshape(atten, (-1, self.interest_num)), 1),
                          dim=-1)  # shape=(batch_size, interest_num)

        if self.hard_readout:  # 选取interest_num个兴趣胶囊中的一个，MIND和ComiRec都是用的这种方式
            readout = torch.reshape(seq_output, (-1, self.hidden_size))[
                (torch.argmax(atten, dim=-1) + torch.arange(label_eb.shape[0]) * self.interest_num).long()]
        else:  # 综合interest_num个兴趣胶囊，论文及代码实现中没有使用这种方法
            readout = torch.matmul(torch.reshape(atten, (label_eb.shape[0], 1, self.interest_num)), seq_output)
            # shape=(batch_size, 1, interest_num)
            # shape=(batch_size, interest_num, hidden_size)
            # shape=(batch_size, 1, hidden_size)
            readout = torch.reshape(readout, (seq_output.shape[0], self.hidden_size))  # shape=(batch_size, hidden_size)
            # scores是vu堆叠成的矩阵（一个batch的vu）（vu可以说就是最终的用户嵌入）

        all_items = self.item_embedding.weight
        scores = torch.matmul(readout, all_items.transpose(1, 0))  # (batch_size, n)
        return scores


# class ComiRec_SA(SequentialRecommender):
#
#     def __init__(self, config, dataset):
#         super(ComiRec_SA, self).__init__(config, dataset)
#
#         # load parameters info
#         # self.ITEM_SEQ = self.ITEM_ID + config['LIST_SUFFIX']
#         # self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
#         # self.embedding_size = config['embedding_size']
#         self.hidden_size = config['hidden_size']  # hidden_size就是embedding_size
#         self.interest_num = config['interest_num']
#         self.num_heads = config['interest_num']
#         self.loss_type = config['loss_type']
#         self.add_pos = config['add_pos']  # bool值, 默认为True
#         self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
#         if self.add_pos:
#             self.position_embedding = nn.Parameter(
#                 torch.Tensor(1, self.max_seq_length, self.hidden_size))  # 论文中max_seq_len是50
#         self.linear1 = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size * 4, bias=False),
#             nn.Tanh()
#         )
#         self.linear2 = nn.Linear(self.hidden_size * 4, self.num_heads, bias=False)
#
#         if self.loss_type == 'BPR':
#             self.loss_fct = BPRLoss()
#         elif self.loss_type == 'CE':
#             self.loss_fct = nn.CrossEntropyLoss()
#         else:
#             raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
#
#         for weight in self.parameters():
#             torch.nn.init.kaiming_normal_(weight)
#
#     def forward(self, item_seq):
#         mask = item_seq.clone()
#         mask[mask != 0] = 1  # batch_size * max_len
#         item_seq_emb = self.item_embedding(item_seq)  # [batch_size, max_len, embedding_size]
#         item_seq_emb = item_seq_emb * torch.reshape(mask, (-1, self.max_seq_length, 1))
#         # mask_shape = batch_size * max_seq_len
#
#         item_seq_emb = torch.reshape(item_seq_emb, (-1, self.max_seq_length, self.hidden_size))
#         if self.add_pos:
#             # 位置嵌入堆叠一个batch，然后历史物品嵌入相加
#             item_seq_emb_pos = item_seq_emb + self.position_embedding.repeat(item_seq_emb.shape[0], 1, 1)
#         else:
#             item_seq_emb_pos = item_seq_emb
#
#         # shape=(batch_size, maxlen, hidden_size*4)
#         item_hidden = self.linear1(item_seq_emb_pos)
#         # shape=(batch_size, maxlen, interest_num)
#         item_att_w = self.linear2(item_hidden)
#         # shape=(batch_size, interest_num, maxlen)
#         item_att_w = torch.transpose(item_att_w, 2, 1).contiguous()
#
#         # shape=(batch_size, interest_num, maxlen)
#         atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.interest_num, 1)
#         paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)  # softmax之后无限接近于0
#
#         item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
#         item_att_w = F.softmax(item_att_w, dim=-1)
#         # 矩阵A，shape=(batch_size, interest_num, maxlen)
#         '''
#         torch.where(condition, x, y)-->Tensor
#         condition是条件，x和y是同shape的矩阵，针对矩阵的某个位置元素，满足条件就返回x，不满足就返回y
#
#         torch.eq(a, b)
#         比较两个矩阵中所有对应位置的元素是否相同
#         举例： torch.eq(a, b)
#               tensor([[False, False, False],
#                     [False, False, False]])
#
#         '''
#
#         # interest_emb,即论文中的部分
#         interest_emb = torch.matmul(item_att_w, item_seq_emb)
#         # item_att_w    shape=(batch_size, interest_num, maxlen)
#         # item_seq_emb  shape=(batch_size, maxlen, embedding_dim) embedding_dim 就是 hidden_size
#         # shape=(batch_size, interest_num, embedding_dim)
#
#         # 用户多兴趣向量
#         user_eb = interest_emb
#         # shape=(batch_size, interest_num, embedding_dim)
#
#         return user_eb
#
#     def calculate_loss(self, interaction):
#         item_seq = interaction[self.ITEM_SEQ]  # 这是一个tensor
#         # item_seq_len = interaction[self.ITEM_SEQ_LEN]  # 这是一个tensor
#
#         '''
#         关于interaction：
#             interaction是一个字典,
#             举个例子：
#             interaction=Interaction({'4': torch.zeros(5), '3': torch.zeros(7)})（Interaction在recbole.data.interaction中）
#             输出：
#             The batch_size of interaction: 7
#                 4, torch.Size([5]), cpu, torch.float32
#                 3, torch.Size([7]), cpu, torch.float32
#
#             interaction['4'] ==>形式等价于上面 interaction[self.ITEM_SEQ]
#             = tensor([0., 0., 0., 0., 0.])
#
#             batch_size取最大值，这里最大的是7，取7【注：batch_size表示第0维】
#
#         '''
#         # seq_output = self.forward(item_seq, item_seq_len)
#         # k = random.choice((range(4, item_seq.shape[0])
#         # k = random.choice((range(4, len(list(item_seq)))))
#
#         seq_output = self.forward(item_seq)
#
#         pos_items = interaction[self.POS_ITEM_ID]  # POS_ITEM_ID就是item_id, 就是labels
#
#         if self.loss_type == 'BPR':
#             neg_items = interaction[self.NEG_ITEM_ID]
#             pos_items_emb = self.item_embedding(pos_items)
#             neg_items_emb = self.item_embedding(neg_items)
#             pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
#             neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
#             loss = self.loss_fct(pos_score, neg_score)
#             return loss
#         else:  # self.loss_type = 'CE'
#             test_item_emb = self.item_embedding.weight
#             logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
#             loss = self.loss_fct(logits.sum(1), pos_items)
#
#         return loss
#
#
# def predict(self, interaction):
#     item_seq = interaction[self.ITEM_SEQ]  # shape=(batch_size,max_len)
#     test_item = interaction[self.ITEM_ID]
#     seq_output = self.forward(item_seq)
#     label_eb = self.item_embedding(test_item)
#     atten = torch.matmul(seq_output,  # shape=(batch_size, interest_num, hidden_size)
#                          torch.reshape(label_eb, (-1, self.hidden_size, 1))
#                          # shape=(batch_size, hidden_size, 1)
#                          )  # shape=(batch_size, interest_num, 1)
#     atten = F.softmax(torch.pow(torch.reshape(atten, (-1, self.interest_num)), 1),
#                       dim=-1)  # shape=(batch_size, interest_num)
#
#     if self.hard_readout:  # 选取interest_num个兴趣胶囊中的一个，MIND和ComiRec都是用的这种方式
#         readout = torch.reshape(seq_output, (-1, self.hidden_size))[
#             (torch.argmax(atten, dim=-1) + torch.arange(label_eb.shape[0]) * self.interest_num).long()]
#     else:  # 综合interest_num个兴趣胶囊，论文及代码实现中没有使用这种方法
#         readout = torch.matmul(torch.reshape(atten, (label_eb.shape[0], 1, self.interest_num)), seq_output)
#         # shape=(batch_size, 1, interest_num)
#         # shape=(batch_size, interest_num, hidden_size)
#         # shape=(batch_size, 1, hidden_size)
#         readout = torch.reshape(readout, (seq_output.shape[0], self.hidden_size))  # shape=(batch_size, hidden_size)
#         # scores是vu堆叠成的矩阵（一个batch的vu）（vu可以说就是最终的用户嵌入）
#
#     scores = readout.sum(dim=1)  # [B]
#
#     return scores
#
#
# def full_sort_predict(self, interaction):  # interaction是一个字典
#     item_seq = interaction[self.ITEM_SEQ]  # shape=(batch_size,max_len)
#     test_item = interaction[self.ITEM_ID]  # batch_size * n
#     # item_seq_len = interaction[self.ITEM_SEQ_LEN]  # shape=(batch_size, seq_len)
#     # k = random.choice((range(4, item_seq.shape[0])
#     k = random.choice(range(4, item_seq.shape[0]))
#     # label_seq = []
#     # mask = []
#     # if k >= item_seq_len:
#     #     label_seq.append(list(item_seq.numpy().keys())[k - item_seq_len:k])
#     #     mask.append(list(item_seq.numpy().keys())[[1.0] * item_seq_len])
#     # else:
#     #     label_seq.append(list(item_seq.numpy().keys())[:k] + [0] * (item_seq_len - k))
#     #     mask.append(list(item_seq.numpy().keys())[[1.0] * k + [0.0] * (item_seq_len - k)])
#     # label_seq = list(map(float, label_seq))
#     # mask = list(map(float, mask))
#     # label_seq = torch.tensor(label_seq)
#     # mask = torch.tensor(mask)
#     seq_output = self.forward(item_seq)
#     # 这里seq_output就是user_eb
#     # 这个模型训练过程中label是可见的，此处的item_eb就是label物品的嵌入
#     label_eb = self.item_embedding(test_item)
#     atten = torch.matmul(seq_output,  # shape=(batch_size, interest_num, hidden_size)
#                          torch.reshape(label_eb, (-1, self.hidden_size, 1))
#                          # shape=(batch_size, hidden_size, 1)
#                          )  # shape=(batch_size, interest_num, 1)
#     atten = F.softmax(torch.pow(torch.reshape(atten, (-1, self.interest_num)), 1),
#                       dim=-1)  # shape=(batch_size, interest_num)
#
#     if self.hard_readout:  # 选取interest_num个兴趣胶囊中的一个，MIND和ComiRec都是用的这种方式
#         readout = torch.reshape(seq_output, (-1, self.hidden_size))[
#             (torch.argmax(atten, dim=-1) + torch.arange(label_eb.shape[0]) * self.interest_num).long()]
#     else:  # 综合interest_num个兴趣胶囊，论文及代码实现中没有使用这种方法
#         readout = torch.matmul(torch.reshape(atten, (label_eb.shape[0], 1, self.interest_num)), seq_output)
#         # shape=(batch_size, 1, interest_num)
#         # shape=(batch_size, interest_num, hidden_size)
#         # shape=(batch_size, 1, hidden_size)
#         readout = torch.reshape(readout, (seq_output.shape[0], self.hidden_size))  # shape=(batch_size, hidden_size)
#         # scores是vu堆叠成的矩阵（一个batch的vu）（vu可以说就是最终的用户嵌入）
#
#     all_items = self.item_embedding.weight
#     scores = torch.matmul(readout, all_items.transpose(1, 0))  # (batch_size, n)
#     return scores
