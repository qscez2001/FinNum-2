import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from transformers.modeling_bert import BertPreTrainedModel

class BertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, bidirectional=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        # self.weights = [10, 1]
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]

        # lstm_out, _ = self.lstm(pooled_output.view(len(token_type_ids), 1, -1))

        pooled_output = self.dropout(pooled_output)
        # pooled_output = self.dropout(lstm_out)

        logits = self.classifier(pooled_output.view(len(token_type_ids), -1))
        # logits = pooled_output

        # outputs = (logits,) + outputs[0:]
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # class_weights=torch.FloatTensor(self.weights).cuda()
            # [0.9, 0.1] [0.99, 0.01] [0.8, 0.2] 
            loss_fct = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.9, 0.1]).cuda())
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs, pooled_output  # (loss), logits, (hidden_states), (attentions), pooled_output
