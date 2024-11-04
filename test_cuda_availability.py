# test_cuda.py
import torch

def check_gpu_details():
    print("\nGPU Details:")
    print("-" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU available - running on CPU")
        print("Checking CUDA installation...")
        try:
            import nvidia.cuda
            print("CUDA toolkit is installed")
        except ImportError:
            print("CUDA toolkit not found")
    print("-" * 40)

check_gpu_details()