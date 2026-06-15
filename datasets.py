import torch

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

        y = sample["output_fields"][0]

        return x.float(), y.float()