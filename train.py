from datasets import load_dataset
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import torch
import torch.nn as nn

class Config():
    def __init__(self):
        self.exp_name = 'wikitext-2-raw-v1'
        self.exp_id = '1'
        self.num_epochs = 10
        self.batch_size = 200
        self.num_workers = 0
        self.max_length = 512
        self.vocab_size = AutoTokenizer.from_pretrained('bert-base-uncased').vocab_size
        self.embedding_dim = 100
        self.hidden_dim = 100
        self.dropout_rate = 0.5
        self.num_layers = 2
        self.tie_weights = True
        self.lr = 1e-3
        self.lr_end_factor = 1e-3

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        print(f'using {self.device}')

        self.base_dir = f'results/{self.exp_name}/{self.exp_id}'
        self.log_dir = f'{self.base_dir}/logs'
        self.model_dir = f'{self.base_dir}/models'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            os.makedirs(self.log_dir)
            os.makedirs(self.model_dir)

class RNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            dropout=config.dropout_rate,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(
            in_features=config.hidden_dim,
            out_features=config.vocab_size,
        )
        self.dropout = nn.Dropout(p=config.dropout_rate)
        if config.tie_weights:
            self.fc.weight = self.embedding.weight
    
    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def collate(max_length, batch):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    batch = [doc['text'] for doc in batch]
    encoded_inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    return encoded_inputs['input_ids']

def train():
    config = Config()

    ds_train = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')['train']
    dl_train = DataLoader(
        dataset=ds_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=partial(collate, config.max_length)
    )

    model = RNN(config)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.lr,
    )
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=1.0,
        end_factor=config.lr_end_factor,
        total_iters=config.num_epochs * len(dl_train)
    )
    logger = SummaryWriter(log_dir=config.log_dir)
    step = 0
    for epoch in tqdm(range(config.num_epochs), 'Epoch: '):
        for batch, token_ids in enumerate(tqdm(dl_train, 'Batch: ')):
            input_ids = token_ids[:,:-1]
            targets = token_ids[:,1:]

            # print(f'batch {i}')
            # print(f'input_ids = {tokenizer.decode(input_ids[0])}')
            # print(f'targets = {tokenizer.decode(targets[0])}')
            # print(f'masks = {masks[0]}')
            # for j in range(input_ids.shape[1]):
            #     if tokenizer.decode(input_ids[0][j]) == '[SEP]':
            #         print(f'input_ids [SEP] index: {j}')
            #         print(f'mask[j] = {masks[0][j]}')
            #         print(f'mask[j-1] = {masks[0][j-1]}')
            #         print(f'target[j] = {tokenizer.decode(targets[0][j])}')
            #         print(f'target[j-1] = {tokenizer.decode(targets[0][j-1])}')
    
            model.train()
            logits = model(input_ids)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            # print(f'logits={logits.shape}')
            # print(f'targets={targets.shape}')

            loss = criterion(logits, targets)
            print(f'Epoch {epoch}, batch {batch}: loss={loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            step += 1
            logger.add_scalar('training_loss', loss.item(), step)
            logger.add_scalar('lr', lr_scheduler.get_last_lr()[0], step)
            logger.flush()

        model.eval()
        torch.save(
            model.load_state_dict(),
            f'{config.model_dir}/checkpoint-{epoch}.pt',
        )

if __name__ == '__main__':
    train()