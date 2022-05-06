import argparse
import logging
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer
from transformers.optimization import AdamW
from torch.utils.data import DataLoader
from model import ptuning
from dataset import UFET
from utils import *


def set_logger(args):
    log_file = os.path.join(args.save_dir, args.dataset, 'log.txt')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def get_params():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', type=str, default='ultra')
    parser.add_argument('--ontology', type=str, default='./data/ontology/types.txt')
    parser.add_argument('--train', type=str, default='./data/UFET/crowd/train.json')
    parser.add_argument('--valid', type=str, default='./data/UFET/crowd/dev.json')
    parser.add_argument('--test', type=str, default='./data/UFET/crowd/test.json')
    parser.add_argument('--label_correction', action='store_true', default=False)
    parser.add_argument('--FP_mask', type=str, default='./save/UFET/FP_mask.pth')
    parser.add_argument('--FN_mask', type=str, default='./save/UFET/FN_mask.pth')
    # model
    parser.add_argument('--backbone', type=str, default='bert-base-cased')
    parser.add_argument('--dense_param', type=str, default='./save/dense.pth')
    parser.add_argument('--ln_param', type=str, default='./save/ln.pth')
    parser.add_argument('--fc_param', type=str, default='./save/ultra_fc.pth')
    # training & test
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--valid_step', type=int, default=500)
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--patience', type=int, default=6)
    # log & model save
    parser.add_argument('--save_dir', type=str, default='./test')

    args, _ = parser.parse_known_args()
    return args


def evaluation(model, use_cuda, data, step):
    model.eval()
    ground_truth = []
    predict = []
    with torch.no_grad():
        for _, input_ids, attention_mask, mask_position, labels in data:
            if use_cuda:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                mask_position = mask_position.cuda()
                labels = labels.cuda()

            temp = model(input_ids, attention_mask, mask_position).sigmoid()
            temp = binarization(temp)
            predict.append(temp)
            ground_truth.append(labels)
        predict = torch.cat(predict, dim=0)
        ground_truth = torch.cat(ground_truth, dim=0)
        _, _, maf1 = record_metrics(step, ground_truth, predict)
        return maf1


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    type2id = dict()
    with open(args.ontology) as f:
        for line in f.readlines():
            type2id[line.strip()] = len(type2id)

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.backbone)
    # add prompt placeholder
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
    prompt_placeholder_id = tokenizer.additional_special_tokens_ids[0]
    unk_id = tokenizer.unk_token_id

    # create dataset
    if args.label_correction:
        train = DataLoader(
            dataset=UFET(args.train, type2id,
                         FP_mask=torch.load(args.FP_mask),
                         FN_mask=torch.load(args.FN_mask)),
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=lambda x: UFET.collate_fn(x, tokenizer),
            drop_last=False,
            num_workers=8
        )
    else:
        train = DataLoader(
            dataset=UFET(args.train, type2id),
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=lambda x: UFET.collate_fn(x, tokenizer),
            drop_last=False,
            num_workers=8
        )
    valid = DataLoader(
        dataset=UFET(args.valid, type2id),
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=lambda x: UFET.collate_fn(x, tokenizer),
        drop_last=False,
        num_workers=8
    )
    test = DataLoader(
        dataset=UFET(args.test, type2id),
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=lambda x: UFET.collate_fn(x, tokenizer),
        drop_last=False,
        num_workers=8
    )

    # model
    model = ptuning(3, len(type2id), prompt_placeholder_id, unk_id, args.backbone,
                    dense_param=args.dense_param, ln_param=args.ln_param, fc_param=args.fc_param)
    if use_cuda:
        model = model.to('cuda')
    # print model params
    # for name, param in model.named_parameters():
    #     logging.info('Parameter %s: %s, require_grad=%s' % (name, str(param.size()), str(param.requires_grad)))

    # optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    # criterion
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # metric
    best_macro_f1 = 0
    step = 0
    kill = 0
    early_stop = False
    log = []

    # training loop
    for epoch in range(args.max_epoch):
        for idx, input_ids, attention_mask, mask_position, labels in train:
            model.train()
            step += 1
            if use_cuda:
                idx = idx.cuda()
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                mask_position = mask_position.cuda()
                labels = labels.cuda()
            predict = model(input_ids, attention_mask, mask_position)
            loss = criterion(predict, labels).mean()

            log.append({
                'loss': loss.item(),
            })

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_step == 0:
                logging.info(f'step{step} avg_loss:{sum([_["loss"] for _ in log]) / len(log)}')
                log = []

            # valid
            if step % args.valid_step == 0:
                macro_f1 = evaluation(model, use_cuda, valid, step)
                if macro_f1 > best_macro_f1:
                    torch.save(model, os.path.join(args.save_dir, args.dataset, 'model.pth'))
                    kill = 0
                    best_macro_f1 = macro_f1
                else:
                    kill += 1
                    if kill == args.patience:
                        early_stop = True
                        break
        if early_stop:
            break

    # test
    model = torch.load(os.path.join(args.save_dir, args.dataset, 'model.pth')).cuda()
    evaluation(model, use_cuda, test, step)


if __name__ == '__main__':
    try:
        params = get_params()
        if not os.path.exists(os.path.join(params.save_dir, params.dataset)):
            os.makedirs(os.path.join(params.save_dir, params.dataset))
        set_logger(params)
        main(params)
    except Exception as e:
        logging.exception(e)
        raise
