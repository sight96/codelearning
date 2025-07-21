import torch
import torch.nn as nn
import math

# 定义位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        print("pe shape:", pe.shape)
        print("pe:\n", pe)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        print("pos shape:", pe.shape)
        print("pos:\n", pe)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        print("div shape:", pe.shape)
        print("div:\n", pe)
        pe[:, 0::2] = torch.sin(position * div_term)
        print("pe shape:", pe.shape)
        print("pe:\n", pe)
        pe[:, 1::2] = torch.cos(position * div_term)
        print("pe shape:", pe.shape)
        print("pe:\n", pe)
        pe = pe.unsqueeze(0)  # 增加批量维度
        print("pe shape:", pe.shape)
        print("pe:\n", pe)
        self.register_buffer('pe', pe)

    def forward(self, x):
        print("输入张量 shape:", x.shape)
        print("输入张量内容:\n", x)
        x = x + self.pe[:, :x.size(1), :]
        print("应用位置编码后的输出形状:", x.shape)
        print("应用位置编码后的输出内容:\n", x)
        return x

# 模拟参数
d_model = 8   # 词嵌入维度
max_len = 4  # 最大序列长度
batch_size = 1  # 批数量
seq_len = 4    # 序列长度

# 随机生成输入张量
input_tensor = torch.rand(batch_size, seq_len, d_model)

# 创建位置编码层
positional_encoding = PositionalEncoding(d_model, max_len)

# 前向传播
output_tensor = positional_encoding(input_tensor)