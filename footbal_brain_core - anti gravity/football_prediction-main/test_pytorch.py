import torch

print("=" * 60)
print("PyTorch Kurulum Kontrolü")
print("=" * 60)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print("\n✅ GPU HAZIR! CUDA ile çalışacak!")
else:
    print("\n⚠️ GPU bulunamadı, CPU ile çalışacak")

print("=" * 60)

