import pandas as pd
import torch
import pytorch_lightning as pl
from torch.nn import functional as f
from torch.utils.data import DataLoader, Dataset, random_split


class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(99, 10)
        self.layer_2 = torch.nn.Linear(10, 128)
        self.layer_3 = torch.nn.Linear(128, 512)
        # 293 - nunique moves
        self.layer_4 = torch.nn.Linear(512, 293)

    def forward(self, x):
        # Embedding layer
        x = self.layer_1(x)

        x = self.layer_2(x)
        x = torch.relu(x)

        x = self.layer_3(x)
        x = torch.relu(x)

        x = self.layer_4(x)
        x = torch.softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return f.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer


class Match3DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, train_eval_split=0.9):
        super().__init__()
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.batch_size = batch_size
        self.train_eval_split = train_eval_split

    def setup(self, stage):
        train_dataset_full = CustomMatch3Dataset('train')
        train_set_size = int(len(train_dataset_full) * self.train_eval_split)
        valid_set_size = len(train_dataset_full) - train_set_size
        self.train_dataset, self.val_dataset = random_split(
            train_dataset_full, [train_set_size, valid_set_size])
        self.test_dataset = CustomMatch3Dataset('test')

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
    def __init__(self, stage):
        data_path = 'data/train_output_v2_1000_rows.parquet'
        if stage == 'test':
            data_path = 'data/test_output_v2_1000_rows.parquet'
        self.df = pd.read_parquet(data_path)
        self.df_labels = self.df[['move_id']]
        self.dataset = (torch.tensor(self.df.drop(columns=['move_id'])
                                     .to_numpy()).float())
        self.labels = torch.tensor(
            self.df_labels.to_numpy().reshape(-1)).long()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]


if __name__ == '__main__':
    data_module = Match3DataModule(batch_size=10)
    model = LightningMNISTClassifier()
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, data_module)
