import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 modes1,
                 modes2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        B, C_in, H, W = x.shape

        x = x.float() 
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=x_ft.dtype,
            device=x.device
        )

        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(H, W))
   
    
class FourierLayer2d(nn.Module):
    def __init__(self,
                 width,
                 modes1,
                 modes2):
        super().__init__()

        self.spectral = SpectralConv2d(
            width,
            width,
            modes1,
            modes2
        )

        self.pointwise = nn.Conv2d(
            width,
            width,
            kernel_size=1
        )

    def forward(self, x):

        return (
            self.spectral(x)
            + self.pointwise(x)
        )
    

class FNO2d(nn.Module):
    def __init__(
        self,
        modes1=16,
        modes2=16,
        width=64,
        in_dim=3,
        out_dim=1,
        depth=4,
        proj_dim=128
    ):
        super().__init__()

        self.lift = nn.Linear(
            in_dim,
            width
        )

        self.layers = nn.ModuleList([
            FourierLayer2d(
                width,
                modes1,
                modes2
            )
            for _ in range(depth)
        ])

        self.proj1 = nn.Linear(
            width,
            proj_dim
        )

        self.proj2 = nn.Linear(
            proj_dim,
            out_dim
        )

    def forward(self, x):
        # (B,nx,ny,in_dim)
        x = self.lift(x)

        # (B,width,nx,ny)
        x = x.permute(
            0, 3, 1, 2
        )

        for layer in self.layers:
            x = layer(x)
            x = F.gelu(x)

        # (B,nx,ny,width)
        x = x.permute(
            0, 2, 3, 1
        )

        x = self.proj1(x)
        x = F.gelu(x)

        x = self.proj2(x)

        return x

class FourierLayer2d_v1(nn.Module):
    def __init__(self,
                 width,
                 modes1,
                 modes2):
        super().__init__()

        self.spectral = SpectralConv2d(
            width,
            width,
            modes1,
            modes2
        )

        self.pointwise = nn.Conv2d(
            width,
            width,
            kernel_size=1
        )

        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x):

        x = self.spectral(x) + self.pointwise(x)

        x = self.norm(x)

        return x

class FNO2d_v1(nn.Module):
    def __init__(
        self,
        modes1=16,
        modes2=16,
        width=64,
        width1=64,
        in_dim=3,
        out_dim=1,
        depth=4,
        proj_dim=128
    ):
        super().__init__()

        self.l1 = nn.Linear(
            in_dim,
            width1
        )

        self.l2 = nn.Linear(
            width1,
            width
        )

        self.layers = nn.ModuleList([
            FourierLayer2d_v1(
                width,
                modes1,
                modes2
            )
            for _ in range(depth)
        ])

        self.proj1 = nn.Linear(
            width,
            proj_dim
        )

        self.proj2 = nn.Linear(
            proj_dim,
            out_dim
        )

    def forward(self, x):
        # (B,nx,ny,in_dim)
        x = self.l1(x)
        x = F.gelu(x)
        x = self.l2(x)

        # (B,width,nx,ny)
        x = x.permute(
            0, 3, 1, 2
        )

        for layer in self.layers:
            x = layer(x)
            x = F.gelu(x)

        # (B,nx,ny,width)
        x = x.permute(
            0, 2, 3, 1
        )

        x = self.proj1(x)
        x = F.gelu(x)

        x = self.proj2(x)

        return x

def train_one_epoch(model, loader, optimizer, criterion, epoch, epochs):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

    for batch_idx, (x, y) in enumerate(pbar):
        optimizer.zero_grad()

        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x)
            loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{running_loss / (batch_idx + 1):.6f}"})

    return running_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x)
                val_loss += criterion(pred, y).item()

    return val_loss / len(loader)


def train_model(model, train_loader, val_loader, optimizer, criterion,
                epochs, checkpoint_dir="checkpoints"):
    history = []

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val = np.inf

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch, epochs)
        scheduler.step()

        torch.cuda.empty_cache()

        val_loss = evaluate(model, val_loader, criterion)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, f"{checkpoint_dir}/best_model.pt")

        print(
            f"Epoch {epoch+1:03d} | "
            f"train = {train_loss:.6f} | "
            f"val = {val_loss:.6f} | "
            f"best_val = {best_val:.6f}"
        )

        history.append({
            "epoch": epoch,
            "train": train_loss,
            "val": val_loss,
            "best_val": best_val
        })
    
    return history