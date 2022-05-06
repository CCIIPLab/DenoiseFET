import torch
import logging


def filter_outlier(data, alpha, mode='pos'):
    mean, std = data.mean(), data.std()
    while True:
        if mode == 'pos':
            mask = data < (mean - alpha * std)
        else:
            mask = data > (mean + alpha * std)
        if mask.sum() == 0:
            break
        else:
            data = data[mask == 0]
    return data


def gaussian(x, m, std):
    return 1/(std) * torch.exp(-((x - m)**2 / (2 * std ** 2)))


def posterior(x, pos_m, pos_std, pos_prior, neg_m, neg_std, neg_prior):
    pos_p = gaussian(x, pos_m, pos_std) * pos_prior
    neg_p = gaussian(x, neg_m, neg_std) * neg_prior
    return pos_p / (pos_p + neg_p)


def entropy_loss(predict):
    predict = torch.clamp(predict, max=0.99, min=0.01)
    loss = - predict * torch.log(predict) - (1 - predict) * torch.log(1 - predict)
    return loss


def binarization(predict):
    max_index = torch.argmax(predict, dim=1)
    for dim, i in enumerate(max_index):
        predict[dim, i] = 1
    predict[predict > 0.5] = 1
    predict[predict != 1] = 0
    return predict


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall)


def record_metrics(step, label, predict):
    sample_num = label.size(0)
    # strict metric: P==R==F1
    correct_num = (torch.abs(predict - label).sum(1) == 0).sum().item()
    acc = correct_num / sample_num

    # micro metric
    micro_p = (label * predict).sum() / predict.sum()
    micro_r = (label * predict).sum() / label.sum()
    micro_f1 = f1(micro_p, micro_r)

    # macro metric
    macro_p = ((label * predict).sum(1) / predict.sum(1)).mean()
    macro_r = ((label * predict).sum(1) / label.sum(1)).mean()
    macro_f1 = f1(macro_p, macro_r)

    logging.info('step %d\tmicro_f1: %f\tmacro_f1: %f\tacc: %f' % (step, micro_f1, macro_f1, acc))
    logging.info(f'step{step}: macro_P: {macro_p}, macro_R: {macro_r}, macro_F: {macro_f1}')
    return macro_p, macro_r, macro_f1
