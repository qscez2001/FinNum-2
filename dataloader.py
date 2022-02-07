from preprocess import preprocess
from preprocess import dev_preprocess
from preprocess import test_preprocess

train_x, val_x, train_y , val_y = preprocess()
dev_x, dev_y = dev_preprocess()
# train_x, train_y= preprocess()
# val_x, val_y = dev_preprocess()
test_x, test_y = test_preprocess()
# print(len(train_x), len(val_x))

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# PRETRAINED_MODEL_NAME = "bert-base-cased" 
PRETRAINED_MODEL_NAME = "bert-large-cased"

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

class TweetsDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, list_IDs, labels, tokenizer):
        self.list_IDs = list_IDs
        self.labels = labels
        self.tokenizer = tokenizer 

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        X = self.list_IDs[idx][0]
        feature = self.list_IDs[idx][1]
        # y = self.labels[idx]      
        label_id = self.labels[idx]
        label_tensor = torch.tensor(label_id)

        word_pieces = ["[CLS]"]
        tokens = self.tokenizer.tokenize(X)
        word_pieces += tokens + ["[SEP]"] 
        len_a = len(word_pieces)  
        # print(word_pieces)

        # 第二個句子的 BERT tokens
        tokens_b = self.tokenizer.tokenize(feature)
        word_pieces += tokens_b + ["[SEP]"]
        len_b = len(word_pieces) - len_a   
        
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, 
                                        dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return len(self.list_IDs)

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    

    label_ids = torch.stack([s[2] for s in samples])
    
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids


def create_dataloader():
    
    trainset = TweetsDataset(train_x, train_y, tokenizer=tokenizer)
    valset = TweetsDataset(val_x, val_y, tokenizer=tokenizer)
    
    BATCH_SIZE = 4
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch, shuffle = True)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

    return trainloader, valloader


def getValLoader():

    testloader = []

    testset = TweetsDataset(val_x, val_y, tokenizer=tokenizer)
    # testloader = DataLoader(testset, batch_size=1, collate_fn=create_mini_batch)
    testloader.append(DataLoader(testset, batch_size=1, collate_fn=create_mini_batch))

    return testloader

def getAllLoader():

    all_loader = []
    X = np.concatenate((train_x, val_x))
    y = np.concatenate((train_y, val_y))

    dataset = TweetsDataset(X, y, tokenizer=tokenizer)

    all_loader.append(DataLoader(dataset, batch_size=1, collate_fn=create_mini_batch))

    return all_loader


def getDevLoader():

    devloader = []

    devset = TweetsDataset(dev_x, dev_y, tokenizer=tokenizer)
    # testloader = DataLoader(testset, batch_size=1, collate_fn=create_mini_batch)
    devloader.append(DataLoader(devset, batch_size=1, collate_fn=create_mini_batch))

    return devloader

def getTestLoader():

    testloader = []

    testset = TweetsDataset(test_x, test_y, tokenizer=tokenizer)
    # testloader = DataLoader(testset, batch_size=1, collate_fn=create_mini_batch)
    testloader.append(DataLoader(testset, batch_size=1, collate_fn=create_mini_batch))

    return testloader

