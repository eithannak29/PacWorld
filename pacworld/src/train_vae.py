# train_vae.py

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from vae import VAE
from torchvision.utils import make_grid

def save_image(tensor, filename, nrow=8):
    grid = make_grid(tensor, nrow=nrow)
    ndarr = grid.permute(1, 2, 0).cpu().numpy()
    plt.imshow(ndarr, cmap='gray')
    plt.axis('off')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device).view(-1, 28*28)
        optimizer.zero_grad()
        recon, mean, log_var = model(data)
        loss = model.loss_function(recon, data, mean, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch {epoch}: Average loss: {avg_loss:.4f}')

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device).view(-1, 28*28)
            recon, mean, log_var = model(data)
            test_loss += model.loss_function(recon, data, mean, log_var).item()

    avg_loss = test_loss / len(test_loader.dataset)
    print(f'Test Epoch {epoch}: Average loss: {avg_loss:.4f}')

    data = data.view(-1, 1, 28, 28)
    recon = recon.view(-1, 1, 28, 28)
    comparison = torch.cat([data[:8], recon[:8]])
    save_image(comparison, f'results/reconstruction_epoch_{epoch}.png')

def main():
    batch_size = 128
    epochs = 10
    latent_dim = 32
    learning_rate = 1e-3
    input_dim = 28 * 28
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)

    torch.save(model.state_dict(), "vae_mnist.pth")
    print("Model saved as vae_mnist.pth")

    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device).view(-1, 28*28)
        recon, _, _ = model(data)
        data = data.view(-1, 1, 28, 28)
        recon = recon.view(-1, 1, 28, 28)
        comparison = torch.cat([data[:8], recon[:8]])
        save_image(comparison, 'results/final_reconstruction.png')
        print("Final reconstructions saved in results/final_reconstruction.png")

if __name__ == "__main__":
    main()
