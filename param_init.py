import torch
import os
from transformers import BertTokenizer, BertForMaskedLM


def init_decoder_weight(ontology_file, MLM_decoder, dataset, save_dir):
    type_token = []
    with open(ontology_file, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.replace('_', ' ')
            line = line.replace('/', ' ')
            type_token.append(tokenizer(line, add_special_tokens=False)['input_ids'])
    type2token = torch.zeros(len(type_token), tokenizer.vocab_size)
    for idx, temp in enumerate(type_token):
        for i in temp:
            type2token[idx, i] = 1 / len(temp)
    fc = torch.nn.Linear(768, len(type_token))
    MLM_decoder_weight = MLM_decoder.weight.data
    MLM_decoder_bias = MLM_decoder.bias.data
    fc.weight.data = type2token @ MLM_decoder_weight
    fc.bias.data = type2token @ MLM_decoder_bias
    torch.save(fc.state_dict(), os.path.join(save_dir, f'{dataset}_fc.pth'))


if __name__ == '__main__':
    save_dir = './save'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # init model params from MLMHead
    model = BertForMaskedLM.from_pretrained('bert-base-cased')
    MLM_decoder = model.get_output_embeddings()
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # transform layer & LayerNorm
    dense = model.cls.predictions.transform.dense.state_dict()
    ln = model.cls.predictions.transform.LayerNorm.state_dict()
    torch.save(dense, os.path.join(save_dir, 'dense.pth'))
    torch.save(ln, os.path.join(save_dir, 'ln.pth'))

    # decoder layer
    init_decoder_weight('./data/ontology/onto_ontology.txt', MLM_decoder, 'onto', save_dir)
    init_decoder_weight('./data/ontology/onto_ontology.txt', MLM_decoder, 'augmented_onto', save_dir)
    init_decoder_weight('./data/ontology/types.txt', MLM_decoder, 'ultra', save_dir)
