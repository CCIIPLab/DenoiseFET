import argparse
import os
import torch
import json


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ultra')
    parser.add_argument('--ontology', type=str, default='./data/ontology/types.txt')
    parser.add_argument('--train', type=str, default='./data/UFET/crowd/train.json')
    parser.add_argument('--FP_mask', type=str, default='./save/denoise/ultra/FP_mask.pth')
    parser.add_argument('--FN_mask', type=str, default='./save/denoise/ultra/FN_mask.pth')
    parser.add_argument('--save_dir', type=str, default='./denoised_data')

    args, _ = parser.parse_known_args()
    return args


def main(args):
    # load ontology
    type2id = dict()
    with open(args.ontology) as f:
        for line in f.readlines():
            type2id[line.strip()] = len(type2id)
    id2type = dict([(type2id[t], t) for t in type2id])

    # load FN/FP mask
    FP_mask = torch.load(args.FP_mask)
    FN_mask = torch.load(args.FN_mask)

    # load original datasets
    with open(args.train, 'r') as f:
        data = [line.strip() for line in f]

    with open(os.path.join(args.save_dir, args.dataset, 'train.json'), 'w') as f:
        for idx, line in enumerate(data):
            sample = json.loads(line)
            label = torch.LongTensor([type2id[foo] for foo in sample['y_str'] if foo in type2id])
            label = torch.zeros(len(type2id)).scatter_(0, label, 1)
            label = torch.where(FN_mask[idx, :] == 1, torch.ones_like(label), label)
            label = torch.where(FP_mask[idx, :] == 1, torch.zeros_like(label), label)
            label = torch.nonzero(label, as_tuple=False)
            sample['y_str'] = [id2type[foo.item()] for foo in label]
            f.write(f'{json.dumps(sample)}\n')


if __name__ == '__main__':
    try:
        params = get_params()
        if not os.path.exists(os.path.join(params.save_dir, params.dataset)):
            os.makedirs(os.path.join(params.save_dir, params.dataset))
        main(params)
    except Exception as e:
        print(e)
        raise
