import torch
from torch.utils.data import DataLoader
from torchfm.dataset import MovieLens1MDataset
from torchfm.model import FactorizationMachineModel

# Tải dataset
dataset = MovieLens1MDataset('./ml-1m/ratings.dat')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Khởi tạo mô hình FM
model = FactorizationMachineModel(field_dims=dataset.field_dims, embed_dim=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

# Huấn luyện mô hình
model.train()
for epoch in range(5):  # Train for 5 epochs
    total_loss = 0
    for i, (fields, target) in enumerate(dataloader):
        optimizer.zero_grad()
        y = model(fields)
        loss = criterion(y, target.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')
