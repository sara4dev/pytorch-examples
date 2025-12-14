import torch
from time import sleep

def main() -> None:
    print_gpu_info()
    tensor_memory()

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

