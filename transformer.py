import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# https://blog.csdn.net/weixin_53004531/article/details/149150090
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码模块

        参数：
        - d_model: 每个位置的编码维度，等于Transformer中每个token的embedding维度
        - max_len: 支持的序列最大长度，默认是5000
        """
        super().__init__()
        # 创建一个全为0的张量，用于存储位置编码，形状为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引，从0到 max_len-1，形状为 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # 计算不同维度对应的位置缩放因子
        # div_term是 (d_model // 2,) 大小的张量，对应偶数维度的指数项2i,其中包含除法变乘法的一个小trick
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 将位置乘以缩放因子，然后取正弦函数赋值到pe的偶数列
        # 对于维度0,2,4,...，使用sin(position / (10000^(2i/d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        # 将位置乘以缩放因子，然后取余弦函数赋值到pe的奇数列
        # 对于维度1,3,5,...，使用cos(position / (10000^(2i/d_model)))
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加一维用于 batch 维度（广播用），形状变为 (1, max_len, d_model)
        # 方便后续加到输入的嵌入向量上（输入 shape 是 (batch_size, seq_len, d_model)）
        pe = pe.unsqueeze(0)  # Add batch dimension
        # 将位置编码张量注册为buffer，不作为模型参数参与训练
        # 它是常量，但在模型保存和加载时会保留下来
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播：将位置编码加到输入嵌入上

        参数：
        - x: 输入的嵌入张量，形状为 (batch_size, seq_len, d_model)

        返回：
        - 加入位置编码后的张量，形状相同 (batch_size, seq_len, d_model)
        """
        # 只取对应序列长度的部分位置编码，加到输入上
        # self.pe[:, :x.size(1), :] 的 shape 是 (1, seq_len, d_model)
        x = x + self.pe[:, :x.size(1),:]
        return x

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)  # 获取键的维度
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # 计算注意力分数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)  # 应用softmax
    output = torch.matmul(attn, v)  # 计算加权和    

