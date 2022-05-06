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
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--valid_step', type=int, default=500)
    parser.add_argument('--stage1_step', type=int, default=2500)
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--filter_on', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--entropy_loss_on', action='store_true', default=False)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--stage2_step', type=int, default=2000)
    # log & model save
    parser.add_argument('--save_dir', type=str, default='./save/denoise')

    args, _ = parser.parse_known_args()
    return args


def training(model,
             optimizer,
             train,
             valid,
             save_file,
             use_cuda,
             stage1,
             args,
             FP_candidate=None,
             FN_candidate=None):
    step = 0
    best_maf1 = 0
    log = []
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    if FP_candidate is not None:
        mask = FP_candidate + FN_candidate
        if use_cuda:
            mask = mask.cuda()

    while True:
        for idx, input_ids, attention_mask, mask_position, labels in train:
            model.train()
            step += 1
            idx = idx.cuda()
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            mask_position = mask_position.cuda()
            labels = labels.cuda()
            predict = model(input_ids, attention_mask, mask_position)

            if FP_candidate is not None:
                batch_mask = mask[idx, :]
                if args.entropy_loss_on:
                    loss = torch.where(batch_mask == 1,
                                       args.beta * entropy_loss(predict.sigmoid()),
                                       criterion(predict, labels)).mean()
                else:
                    loss = torch.where(batch_mask == 1,
                                       torch.zeros_like(predict),
                                       criterion(predict, labels)).mean()
            else:
                loss = criterion(predict, labels).mean()

            log.append({
                'loss': loss.item(),
            })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.log_step == 0:
                logging.info(f'step{step}, avg_loss:{sum([_["loss"] for _ in log]) / len(log)}')
                log = []

            if stage1:
                if args.eval and step % args.valid_step == 0:
                    maf1 = evaluation(model, use_cuda, valid, step)
                    if maf1 > best_maf1:
                        torch.save(model, save_file)
                        best_maf1 = maf1
                    else:
                        return
                if not args.eval and step == args.stage1_step:
                    torch.save(model, save_file)
                    return
            else:
                if step == args.stage2_step:
                    torch.save(model, save_file)
                    return


def infer(model_path, data, use_cuda, args):
    model = torch.load(model_path)
    model.eval()
    predict = torch.zeros(len(data.dataset), args.label_num)
    if use_cuda:
        predict = predict.cuda()
        model.to('cuda')
    with torch.no_grad():
        for idx, input_ids, attention_mask, mask_position, _ in data:
            if use_cuda:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                mask_position = mask_position.cuda()
            predict[idx, :] = model(input_ids, attention_mask, mask_position)
    return predict.cpu()


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

            temp = model(input_ids, attention_mask, mask_position)
            temp = binarization(temp.sigmoid())
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
    args.label_num = len(type2id)

    # stage0: store original label
    original_label = torch.zeros(len(train.dataset), args.label_num)
    for idx, _, _, _, labels in train:
        original_label[idx, :] = labels
    torch.save(original_label, os.path.join(args.save_dir, args.dataset, 'original_label.pth'))

    # stage1: pretraining and get ambiguous samples
    # define model & optimizer
    model = ptuning(3, len(type2id), prompt_placeholder_id, unk_id, args.backbone,
                    dense_param=args.dense_param, ln_param=args.ln_param, fc_param=args.fc_param)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    if use_cuda:
        model = model.to('cuda')
    stage1_model_path = os.path.join(args.save_dir, args.dataset, 'stage1.pth')
    training(model, optimizer, train, valid, stage1_model_path, use_cuda, True, args)
    stage1_predict = infer(os.path.join(args.save_dir, args.dataset, 'stage1.pth'), train, use_cuda, args)
    torch.save(stage1_predict, os.path.join(args.save_dir, args.dataset, 'stage1_predict.pth'))

    # get noisy candidate
    stage1_predict = torch.load(os.path.join(args.save_dir, args.dataset, 'stage1_predict.pth'))
    FN_candidate = torch.zeros_like(stage1_predict)
    FP_candidate = torch.zeros_like(stage1_predict)
    for i in range(original_label.size(1)):
        p = stage1_predict[:, i]
        l = original_label[:, i]
        if l.sum() == 0:
            continue
        if args.filter_on:
            pos = filter_outlier(p[l == 1], args.alpha)
            neg = filter_outlier(p[l == 0], args.alpha, mode='neg')
        else:
            pos = p[l == 1]
            neg = p[l == 0]
        pos_prior = pos.numel() / (pos.numel() + neg.numel())
        neg_prior = 1 - pos_prior
        pos_m, pos_std = pos.mean(), pos.std()
        neg_m, neg_std = neg.mean(), neg.std()
        temp = posterior(p, pos_m, pos_std, pos_prior, neg_m, neg_std, neg_prior)
        FN_candidate[:, i] = ((temp > 0.5 - args.threshold) * (l == 0)).float()
        FP_candidate[:, i] = ((temp < 0.5 + args.threshold) * (l == 1)).float()
    torch.save(FN_candidate, os.path.join(args.save_dir, args.dataset, 'FN_candidate.pth'))
    torch.save(FP_candidate, os.path.join(args.save_dir, args.dataset, 'FP_candidate.pth'))
    logging.info(f'FN_candidate: {FN_candidate.sum()} FP_candidate: {FP_candidate.sum()}')

    # stage2: train on clean
    # define data & model & optimizer
    train = DataLoader(
        dataset=UFET(args.train, type2id),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda x: UFET.collate_fn(x, tokenizer),
        drop_last=False,
        num_workers=8
    )
    model = torch.load(os.path.join(args.save_dir, args.dataset, 'stage1.pth'))
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    if use_cuda:
        model.to('cuda')
    stage2_model_path = os.path.join(args.save_dir, args.dataset, 'stage2.pth')
    training(model, optimizer, train, valid, stage2_model_path, use_cuda, False, args, FP_candidate, FN_candidate)
    stage2_predict = infer(stage2_model_path, train, use_cuda, args)
    torch.save(stage2_predict, os.path.join(args.save_dir, args.dataset, 'stage2_predict.pth'))
    FN_mask = ((stage2_predict > 0) * (FN_candidate == 1)).float()
    FP_mask = ((stage2_predict < 0) * (FP_candidate == 1)).float()
    torch.save(FN_mask.cpu(), os.path.join(args.save_dir, args.dataset, 'FN_mask.pth'))
    torch.save(FP_mask.cpu(), os.path.join(args.save_dir, args.dataset, 'FP_mask.pth'))
    logging.info(f'FN: {FN_mask.sum()} FP: {FP_mask.sum()}')


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
