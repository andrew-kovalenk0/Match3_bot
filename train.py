import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split


class Match3Bot(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.inner = torch.nn.Sequential(
            torch.nn.Embedding(73, 5),
            torch.nn.Flatten(),
            torch.nn.Linear(495, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 294),
            torch.nn.Sigmoid())

    def forward(self, x):
        return self.inner(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x.to(torch.int64))
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        self.log('Training loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x.to(torch.int64))
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        self.log('Validation loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class Match3DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, train_eval_split=0.8,
                 train_test_split=0.9):
        super().__init__()
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.batch_size = batch_size
        self.train_eval_split = train_eval_split
        self.train_test_split = train_test_split

    def setup(self, stage):
        full_dataset = CustomMatch3Dataset()
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [-1 + self.train_eval_split + self.train_test_split,
                           1 - self.train_eval_split,
                           1 - self.train_test_split])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=12)


class CustomMatch3Dataset(Dataset):
    def __init__(self):
        data_path = f'data/output_v2_5000000.parquet'
        self.df = pd.read_parquet(data_path)
        self.df_labels = self.df[['move_id']]
        self.dataset = (torch.tensor(self.df.drop(columns=['move_id'])
                                     .to_numpy()).int())
        self.labels = torch.tensor(
            self.df_labels.to_numpy().reshape(-1)).long()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]


if __name__ == '__main__':
    data_module = Match3DataModule()
    model = Match3Bot(lr=1e-4)
    trainer = pl.Trainer(max_epochs=200)
    trainer.fit(model, data_module)
