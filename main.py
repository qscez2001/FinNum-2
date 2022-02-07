from dataloader import create_dataloader
from model import BertForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import BertTokenizer
import torch

# PRETRAINED_MODEL_NAME = "bert-base-cased"
PRETRAINED_MODEL_NAME = "bert-large-cased"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

def main():
    torch.manual_seed(0)
    
    # 載入一個可以做中文多分類任務的模型，n_class = 3
    # PRETRAINED_MODEL_NAME = "bert-base-cased"
    PRETRAINED_MODEL_NAME = "bert-large-cased"
    NUM_LABELS = 2
    
    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS, output_hidden_states=True)

    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    train(model)


def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    pooled_outputs = None
    correct = 0
    total = 0
      
    with torch.no_grad():
        
        for data in dataloader:
            
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            tokens_tensors, segments_tensors, masks_tensors = data[:3]        
            tokens = tokens_tensors.cpu().numpy()
            tokens = tokens.flatten()
            tokens = tokenizer.convert_ids_to_tokens(ids=tokens)
            # print(tokens)
            # print(len(tokens))

            outputs, pooled_output = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)

            # print(pooled_output.shape)
            logits = outputs[0]
            # print(outputs[1].shape)

            _, pred = torch.max(logits.data, 1)
            
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
            
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))

            if pooled_outputs is None:
                pooled_outputs = pooled_output
            else:
                pooled_outputs = torch.cat((pooled_outputs, pooled_output), 0)
    
    if compute_acc:
        acc = correct / total
        return predictions, acc
    # else
    return predictions, pooled_outputs


def train(model):
    torch.manual_seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)

    trainloader, valloader = create_dataloader()


    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


    EPOCHS = 5
    for epoch in range(EPOCHS):
        
        running_loss = 0.0
        for data in trainloader:
            
            tokens_tensors, segments_tensors, \
            masks_tensors, labels = [t.to(device) for t in data]

            optimizer.zero_grad()
            
            outputs, pooled_outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors, 
                            labels=labels)

            loss = outputs[0]

            loss.backward()
            optimizer.step()

            # 紀錄當前 batch loss
            running_loss += loss.item()
            
        # 計算分類準確率
        _, train_acc = get_predictions(model, trainloader, compute_acc=True)
        _, val_acc = get_predictions(model, valloader, compute_acc=True)
        print('[epoch %d] loss: %.3f, train_acc: %.3f, val_acc: %.3f' %
              (epoch + 1, running_loss, train_acc, val_acc))

    torch.save(model.state_dict(), "modelmodel8__nolstm")



if __name__ == "__main__": main()

