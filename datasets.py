import torch
import h5py

class HelmholtzDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]

        x = torch.cat([
            sample["input_fields"][0],
            sample["constant_fields"],
            sample["space_grid"]
        ], dim=-1)

        y = sample["output_fields"][0, :, :, :]

        return x.float(), y.float()
    

class TRMDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]

        x0 = sample["input_fields"][0]  # (128,384,4)

        H, W = x0.shape[:2]

        scalar_channel = (
            sample["constant_scalars"]
            .view(1, 1, 1)
            .expand(H, W, 1)
        )

        x = torch.cat([
            x0,
            scalar_channel,
            sample["space_grid"]
        ], dim=-1)

        y = sample["output_fields"][0]

        return x.float(), y.float()
    
class NStokesDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        ds = h5py.File(path, 'r')

        self.a = ds['a']
        self.u = ds['u']
        self.t = ds['t']

        self.n = self.a.shape[-1]

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):

        a = torch.tensor(
            self.a[:, :, idx],
            dtype = torch.float32
        )

        u = torch.tensor(
            self.u[:, :, :, idx],
            dtype = torch.float32
        )

        return a, u
