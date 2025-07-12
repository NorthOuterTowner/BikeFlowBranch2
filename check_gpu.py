import torch
import torch_geometric
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"PyG Version: {torch_geometric.__version__}")
x = torch.randn(86, 2).cuda()
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).cuda()
edge_weight = torch.ones(2).cuda()
conv = torch_geometric.nn.GCNConv(2, 2).cuda()
out = conv(x, edge_index, edge_weight)
print("GCNConv on GPU successful!")