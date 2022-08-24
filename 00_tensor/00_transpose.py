import torch

tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

# torch.matmul(tensor_A, tensor_B) -> shape ERROR

# View tensor_A and tensor_B
print(tensor_A)
print(tensor_B)

print(tensor_A)

print(tensor_B.T) # way 1
print(torch.transpose(tensor_B,0,1)) # way 2
# print(torch.transpose(tensor_B,3,2)) -> 아님

#########################################################################################################################
# We can make matrix multiplication work between `tensor_A` and `tensor_B` by making their inner dimensions match.

# One of the ways to do this is with a **transpose** (switch the dimensions of a given tensor).

# You can perform transposes in PyTorch using either:
# * `torch.transpose(input, dim0, dim1)` - where `input` is the desired tensor to transpose and `dim0` and `dim1` are the dimensions to be swapped.
# * `tensor.T` - where `tensor` is the desired tensor to transpose.

# Let's try the latter.