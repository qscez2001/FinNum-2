from model import BertForSequenceClassification
from main import get_predictions
from dataloader import getValLoader
from sklearn.metrics import f1_score
from sklearn import metrics
import torch
import numpy as np
from preprocess import preprocess
train_x, val_x, train_y , val_y = preprocess()

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # PRETRAINED_MODEL_NAME = "bert-base-cased"
    PRETRAINED_MODEL_NAME = "bert-large-cased"
    NUM_LABELS = 2

    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

    testloader = getValLoader()

    model.load_state_dict(torch.load("modelmodel8__nolstm", map_location = device))

    model.to(device)

    
    for dataloader in testloader:

        predictions, _ = get_predictions(model, dataloader, compute_acc=False)


    predictions = predictions.cpu().numpy()
    
    # predictions = [1] * 1597
    # predictions = np.asarray(predictions)

    # print(val_y.shape)
    # print(predictions.shape)

    print(f1_score(val_y, predictions, average='macro'))
    print(metrics.confusion_matrix(val_y, predictions))
    print(metrics.classification_report(val_y, predictions, digits=3))
    

if __name__ == "__main__": main()