import torch
import torch.nn as nn
from transformers import BertModel


class ptuning(nn.Module):
    def __init__(self,
                 prompt_num,
                 type_num,
                 prompt_placeholder_id,
                 unk_id,
                 backbone='bert-base-cased',
                 embedding_dim=768,
                 dense_param='./data/PLM_weight/dense.pkl',
                 ln_param='./data/PLM_weight/ln.pkl',
                 fc_param='./data/PLM_weight/fc.pkl',
                 init_template=None):
        super(ptuning, self).__init__()
        self.bert = BertModel.from_pretrained(backbone)
        self.type_num = type_num
        self.embedding_dim = embedding_dim
        self.prompt_encoder = prompt(prompt_num, self.embedding_dim, self.bert.get_input_embeddings(),
                                     prompt_placeholder_id, unk_id, init_template)
        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.ln = nn.LayerNorm(self.embedding_dim, elementwise_affine=True)
        self.fc = nn.Linear(self.embedding_dim, type_num, bias=True)
        self.init_weight(dense_param, ln_param, fc_param)

    def init_weight(self, dense_param, ln_param, fc_param):
        self.dense.load_state_dict(torch.load(dense_param))
        self.ln.load_state_dict(torch.load(ln_param))
        self.fc.load_state_dict(torch.load(fc_param))

    def forward(self, input_ids, attention_mask, mask_position):
        mask_position = mask_position.unsqueeze(1).repeat(1, 1, self.embedding_dim)
        input_embeds = self.prompt_encoder(input_ids)
        predict, mask_output = self.predict_on_mask(input_embeds, attention_mask, mask_position)
        return predict

    def predict_on_mask(self, input_embeds, attention_mask, mask_position):
        output = self.bert(inputs_embeds=input_embeds, attention_mask=attention_mask)[0]
        mask_output = torch.gather(output, dim=1, index=mask_position).squeeze(1)
        mask_output = self.ln(self.dense(mask_output))
        predict = self.fc(mask_output)
        return predict, mask_output


class prompt(nn.Module):
    def __init__(self, prompt_length, embedding_dim, bert_embedding, prompt_placeholder_id, unk_id, init_template=None):
        super(prompt, self).__init__()
        self.prompt_length = prompt_length
        self.embedding_dim = embedding_dim
        self.bert_embedding = bert_embedding
        self.prompt_placeholder_id = prompt_placeholder_id
        self.unk_id = unk_id
        self.prompt = nn.Parameter(torch.randn(prompt_length, embedding_dim))
        if init_template is not None:
            print(init_template)
            self.prompt = nn.Parameter(self.bert_embedding(init_template).clone())

    def forward(self, input_ids):
        bz = input_ids.shape[0]
        raw_embedding = input_ids.clone()
        raw_embedding[raw_embedding == self.prompt_placeholder_id] = self.unk_id
        raw_embedding = self.bert_embedding(raw_embedding)  # (bz, len, embedding_dim)

        prompt_idx = torch.nonzero(input_ids == self.prompt_placeholder_id, as_tuple=False)
        prompt_idx = prompt_idx.reshape(bz, self.prompt_length, -1)[:, :, 1]    # (bz, prompt_len)
        for b in range(bz):
            for i in range(self.prompt_length):
                raw_embedding[b, prompt_idx[b, i], :] = self.prompt[i, :]
        return raw_embedding
