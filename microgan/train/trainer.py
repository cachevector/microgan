import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from .dcgan import MicroDCGANGenerator, MicroDCGANDiscriminator

class MicroGANTrainer:
    def __init__(self, latent_dim=32, channels=1, lr=0.0002, beta1=0.5, beta2=0.999):
        self.latent_dim = latent_dim
        self.channels = channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.generator = MicroDCGANGenerator(latent_dim, channels).to(self.device)
        self.discriminator = MicroDCGANDiscriminator(channels).to(self.device)
        
        self.adversarial_loss = nn.BCELoss()
        
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    def train(self, data_loader, epochs=10, checkpoint_dir="checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(epochs):
            for i, (imgs, _) in enumerate(data_loader):
                batch_size = imgs.shape[0]
                real_imgs = imgs.to(self.device)
                
                # Ground truths
                valid = torch.ones(batch_size, 1, device=self.device)
                fake = torch.zeros(batch_size, 1, device=self.device)
                
                # --- Train Generator ---
                self.optimizer_G.zero_grad()
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                gen_imgs = self.generator(z)
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
                g_loss.backward()
                self.optimizer_G.step()
                
                # --- Train Discriminator ---
                self.optimizer_D.zero_grad()
                real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()
                
                if i % 100 == 0:
                    print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(data_loader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
            
            # Save checkpoint
            torch.save(self.generator.state_dict(), os.path.join(checkpoint_dir, f"generator_epoch_{epoch}.pt"))

        print("Training complete.")
        return self.generator

def create_dummy_dataset(num_samples=1000, channels=1, size=32):
    # Dummy dataset of noise-like images
    imgs = torch.randn(num_samples, channels, size, size)
    labels = torch.zeros(num_samples)
    return DataLoader(TensorDataset(imgs, labels), batch_size=32, shuffle=True)
