import torch
from time import sleep
from jaxtyping import Float

def main() -> None:
    # print_gpu_info()
    # tensor_memory()
    # tensor_elementwise_operations()
    # tensor_matmul()
    # tensor_matrix_transpose()
    einsum()

def einsum() -> None:
    print("-" * 100)
    print("Einsum:")
    print("-" * 100)
    # Batched matmul rule: (..., n, m) @ (..., m, p) -> (..., n, p)
    x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones((2, 3, 4))
    y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones((2, 3, 4))

    # old way
    z = x @ y.transpose(-2, -1)
    print("z: ", z)
    print("z.shape: ", z.shape)

    # new way
    z = torch.einsum(x,y,"batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
    print("z: ", z)
    print("z.shape: ", z.shape)

def tensor_matrix_transpose() -> None:
    print("-" * 100)
    print("Tensor Matrix Transpose:")
    print("-" * 100)
    x = torch.tensor([[1,2,3], [4,5,6]], device="cuda", dtype=torch.float16)
    print("x: ", x)
    print("x.shape: ", x.shape)
    print("x.T: ", x.T)  # matrix transpose (view when possible)
    print("x.t(): ", x.t())  # matrix transpose (2D only)
    print("x.transpose(0, 1): ", x.transpose(0, 1))  # swap dims for any rank
    # Note: x.transpose(0, 2) is invalid here because x is 2D (dims are only 0 and 1).
    x3 = torch.arange(2 * 3 * 4, device="cuda").reshape(2, 3, 4)
    print("x3.shape: ", x3.shape)
    print("x3.transpose(0, 2).shape: ", x3.transpose(0, 2).shape)

def tensor_matmul() -> None:
    print("-" * 100)
    print("Tensor Matrix Multiplication:")
    print("-" * 100)
    x = torch.tensor([[1,2,3], [4,5,6]], device="cuda", dtype=torch.float16)
    y = torch.tensor([[7,8,9], [10,11,12]], device="cuda", dtype=torch.float16)
    print("x * y: ", x * y)
    print("x + y: ", x + y)
    x = torch.ones(4,8,16,32)
    w = torch.ones(32,2)
    y = x @ w
    print("y.shape: ", y.shape)

def tensor_elementwise_operations() -> None:
    print("-" * 100)
    print("Tensor Elementwise Operations:")
    print("-" * 100)
    x = torch.tensor([1,2,3,4], device="cuda", dtype=torch.float16)
    print("x * 3 ", x * 3)
    print("x + x: ", x + x)

def tensor_memory() -> None:
    print("-" * 100)
    print("Tensor Memory:")
    print("-" * 100)
    x = torch.zeros(4,8, device="cuda", dtype=torch.float16)
    print(x)
    print("x.shape: ", x.shape)
    print("x.dtype: ", x.dtype)
    print("x.device: ", x.device)
    print("x.numel: ", x.numel())
    print("x.element_size: ", x.element_size())
    print("x.numel() * x.element_size(): ", x.numel() * x.element_size())

def print_gpu_info() -> None:
    print("-" * 100)
    print("GPU Information:")
    print("-" * 100)
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} properties: {torch.cuda.get_device_properties(i)}")


if __name__ == "__main__":
    main()

