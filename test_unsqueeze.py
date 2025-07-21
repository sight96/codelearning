import torch

# 创建一个一维张量
x = torch.tensor([1, 2, 3, 4])
print("Original tensor:", x)
print("Original shape:", x.shape)  # 输出: torch.Size([4])

# 在第 0 维上增加一个维度
x_unsqueezed_0 = x.unsqueeze(0)
print("After unsqueeze(0):", x_unsqueezed_0)
print("New shape:", x_unsqueezed_0.shape)  # 输出: torch.Size([1, 4])

# 在第 1 维上增加一个维度
x_unsqueezed_1 = x.unsqueeze(1)
print("After unsqueeze(1):", x_unsqueezed_1)
print("New shape:", x_unsqueezed_1.shape)  # 输出: torch.Size([4, 1])

# 在第 2 维上增加一个维度
x_unsqueezed_2 = x_unsqueezed_1.unsqueeze(2)
print("After unsqueeze(2):", x_unsqueezed_2)
print("New shape:", x_unsqueezed_2.shape)  # 输出: torch.Size([4, 1, 1])
