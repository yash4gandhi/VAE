import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch.optim as optim
import os

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

data = MNIST('./mnist_data', transform=transform, download=True)

loader = DataLoader(dataset=data, batch_size=100, shuffle=True, num_workers=20)

class VAE(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to('cuda')      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
    
model = VAE().to('cuda')
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    total_loss = reproduction_loss + KLD
    return total_loss, reproduction_loss, KLD

def train(model, optimizer, epochs, x_dim=784):
    model.train()
    total_losses = []
    reproduction_losses = []
    KLD_losses = []
    
    for epoch in range(epochs):
        overall_loss = 0
        reproduction_epoch_loss = 0
        KLD_epoch_loss = 0

        for batch_idx, (x, _) in enumerate(loader):
            x = x.view(100, x_dim).to('cuda')

            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss, reproduction_loss, KLD = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            reproduction_epoch_loss += reproduction_loss.item()
            KLD_epoch_loss += KLD.item()
            
            loss.backward()
            optimizer.step()

        # Calculate average losses for the epoch
        avg_total_loss = overall_loss / len(loader.dataset)
        avg_reproduction_loss = reproduction_epoch_loss / len(loader.dataset)
        avg_KLD_loss = KLD_epoch_loss / len(loader.dataset)
        
        total_losses.append(avg_total_loss)
        reproduction_losses.append(avg_reproduction_loss)
        KLD_losses.append(avg_KLD_loss)
        
        print(f"Epoch {epoch + 1}, Total Loss: {avg_total_loss:.2f}, Reproduction Loss: {avg_reproduction_loss:.2f}, KLD: {avg_KLD_loss:.2f}")
        
    return total_losses, reproduction_losses, KLD_losses

filename='mnist_autoencoder.pt'
if os.path.isfile(filename):
    model.load_state_dict(torch.load(filename))
    model=model.to('cuda')
else:
    total_losses, reproduction_losses, KLD_losses = train(model, optimizer, epochs=20)
    torch.save(model.state_dict(), filename)

plt.figure(figsize=(10, 5))
plt.plot(total_losses, label="Total Loss")
plt.plot(reproduction_losses, label="Reproduction Loss")
plt.plot(KLD_losses, label="KLD")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Training Losses")
plt.legend()
plt.savefig('vae_losses.png')
# plt.show()
plt.close()

MEAN=.2
LOG_VAR=0

z_sample = torch.tensor([[MEAN,LOG_VAR]],dtype=torch.float32).to('cuda')
x_decoded = model.decode(z_sample)
digit = x_decoded.detach().cpu().reshape(28,28)
plt.imshow(digit, cmap='gray')
plt.axis('off')
plt.savefig('test.jpg')

# plt.show()
plt.close()

def plot_latent_space(model, scale=1.0, n=25, digit_size=28, figsize=15):
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid 
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to('cuda')
            x_decoded = model.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Visualization')
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig('Latent_Space.jpg')
    # plt.show()
    plt.close()


plot_latent_space(model)