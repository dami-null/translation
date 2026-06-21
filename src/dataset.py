import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from src.config import DATA_DIR, BATCH_SIZE


class TranslationDataset(Dataset):

    def __init__(self, data_file: str):
        # [{'en':[1,2,3,4,5],'zh':[1,2,3,4,5]},{'en':[1,2,3,4,5],'zh':[1,2,3,4,5]},{'en':[1,2,3,4,5],'zh':[1,2,3,4,5]}]
        self.data = pd.read_json(data_file, orient='records', lines=True).to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        encoder_input = torch.tensor(line['zh'], dtype=torch.long)
        decoder_input = torch.tensor(line['en'][:-1], dtype=torch.long)
        label = torch.tensor(line['en'][1:], dtype=torch.long)

        return encoder_input, decoder_input, label


def collate_fn(batch):
    """
    整理批数据
    :param batch: 批数据
    :return: encoder_inputs_tensor, decoder_inputs_tensor, labels_tensor
    """
    encoder_inputs = [example[0] for example in batch]
    decoder_inputs = [example[1] for example in batch]
    labels = [example[2] for example in batch]

    # pad_sequence 填充序列 padding_value：填充的值 batch_first：堆叠的第几维
    encoder_inputs_tensor = pad_sequence(encoder_inputs, batch_first=True, padding_value=0)
    decoder_inputs_tensor = pad_sequence(decoder_inputs, batch_first=True, padding_value=0)
    labels_tensor = pad_sequence(labels, batch_first=True, padding_value=0)

    return encoder_inputs_tensor, decoder_inputs_tensor, labels_tensor


def get_dataloader(train: bool = True):
    data_file = str(DATA_DIR / 'processed' / ('train.jsonl' if train else 'test.jsonl'))
    dataset = TranslationDataset(data_file)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


if __name__ == '__main__':
    train_loader = get_dataloader(train=True)
    for encoder_inputs, decoder_inputs, labels in train_loader:
        print(encoder_inputs, decoder_inputs, labels)
        break
