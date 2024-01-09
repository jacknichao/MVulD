import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


class DefectModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DefectModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.args = args

    def get_xcode_vec(self, source_ids):
        mask = source_ids.ne(self.config.pad_token_id)
        token_embeddings = self.encoder(source_ids, attention_mask=mask.unsqueeze(1) * mask.unsqueeze(2))[0]
        sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
        return token_embeddings, sentence_embeddings

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.args.max_source_length)

        _, vec = self.get_xcode_vec(source_ids)

        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

    def get_repr(self, input_ids, labels=None):
        source_ids = input_ids.view(-1, self.args.max_source_length)

        _, vec = self.get_xcode_vec(source_ids)
        return vec, labels
