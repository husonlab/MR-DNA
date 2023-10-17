import sys
import os
from transformers import DistilBertModel, DistilBertForTokenClassification, DistilBertPreTrainedModel, BertModel
from transformers import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
from transformers.modeling_outputs import TokenClassifierOutput
from torch import nn
from torch.nn import CrossEntropyLoss
import torch
from torchcrf import CRF
from methyl_loss import MethyLoss
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, matthews_corrcoef, confusion_matrix


def compute_metrics(labels, preds):
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro')
    cm = confusion_matrix(labels, preds)
    return {
        'acc': acc,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'confusion matrix': cm
    }

# early stopping (f1)
class EarlyStopperF1:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_f1 = 0

    def early_stop(self, validation_f1):
        if validation_f1 > self.max_validation_f1:
            self.max_validation_f1 = validation_f1
            self.counter = 0
        elif validation_f1 < (self.max_validation_f1 - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class DistilBertCRF_MethyLoss(nn.Module):

    def __init__(self, distilbert, num_labels):
        super(DistilBertCRF_MethyLoss, self).__init__()
        self.distilbert = distilbert
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)
        self.focal_loss_gamma=0
        self.focal_loss_alpha=None
        self.num_labels = num_labels

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            CG_annotation=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=None,
    ):

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = outputs[-1][-1]
        output_1 = self.dropout(hidden_state)
        output_2 = self.classifier(output_1)
        logits = output_2

        tags = self.crf.decode(logits)
        tags = torch.Tensor(tags).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if labels is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #[bs, 50, 3] --> [bs*50, 3]
                active_labels = labels.view(-1)[active_loss]
                active_weight = CG_annotation.view(-1)[active_loss]
                loss = MethyLoss(gamma=self.focal_loss_gamma,alpha=self.focal_loss_alpha, size_average=True)(active_logits, active_labels, active_weight)
            else:
                loss = MethyLoss(gamma=self.focal_loss_gamma,alpha=self.focal_loss_alpha, size_average=True)(logits.view(-1, self.num_labels), labels.view(-1), CG_annotation.view(-1))
            return loss, tags
        else:
            return tags


class DistilBertCRF_Focal(nn.Module):

    def __init__(self, distilbert, num_labels):
        super(DistilBertCRF_Focal, self).__init__()
        self.distilbert = distilbert
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)
        self.focal_loss_gamma=5
        self.focal_loss_alpha=[0.1,0.7,0.2]
        self.num_labels = num_labels

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=None,
    ):

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = outputs[-1][-1]
        output_1 = self.dropout(hidden_state)
        output_2 = self.classifier(output_1)
        logits = output_2

        tags = self.crf.decode(logits)
        tags = torch.Tensor(tags).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if labels is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #[bs, 50, 3] --> [bs*50, 3]
                active_labels = labels.view(-1)[active_loss]
                loss = FocalLoss(gamma=self.focal_loss_gamma,alpha=self.focal_loss_alpha)(active_logits, active_labels)
            else:
                loss = FocalLoss(gamma=self.focal_loss_gamma,alpha=self.focal_loss_alpha, size_average=True)(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, tags
        else:
            return tags


class DistilBertCRF(nn.Module):

    def __init__(self, distilbert, num_labels):
        super(DistilBertCRF, self).__init__()
        self.distilbert = distilbert
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=None,
    ):

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = outputs[-1][-1]
        output_1 = self.dropout(hidden_state)
        output_2 = self.classifier(output_1)
        logits = output_2

        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.byte(), reduction='mean')
            tags = self.crf.decode(logits)
            tags = torch.Tensor(tags).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            return loss, tags
        else:
            tags = self.crf.decode(logits)
            tags = torch.Tensor(tags).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            return tags

