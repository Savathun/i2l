import argparse
import logging
import os
import random

import Levenshtein
import numpy
import torch
import wandb
import yaml
from munch import Munch
from torchtext.data import metrics
from tqdm.auto import tqdm

import dataset
from dataset import I2LDataset # keep, needed when loading pickles
import utils
from model import get_model, Model
from utils import parse_args, check_mem, tok2str, post_process, PAD, BOS, EOS

LR_STEP = 30


def train(args):
    dataloader, valdataloader = dataset.load(args.data), dataset.load(args.valdata)
    dataloader.update(**args, test=False)
    valargs = args.copy()
    valargs.update(batchsize=args.testbatchsize, keep_smaller_batches=True, test=True)
    valdataloader.update(**valargs)
    device = args.device
    model = get_model(args)
    if torch.cuda.is_available() and not args.no_cuda:
        check_mem(model, args)
    max_bleu, max_token_acc = 0, 0
    out_path = os.path.join("model/checkpoints", args.name)
    os.makedirs(out_path, exist_ok=True)

    if args.load_ckpt:
        model.load_state_dict(torch.load(args.load_ckpt, map_location=device))

    def save_models(e, step=0):
        torch.save(model.state_dict(), os.path.join(out_path, f'{args.name}_e{e + 1:02d}_step{step:02d}.pth'))
        yaml.dump(dict(args), open(os.path.join(out_path, 'config.yaml'), 'w+'))

    opt = torch.optim.Adam(model.parameters(), args.lr, betas=args.betas)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_STEP, gamma=args.gamma)

    micro_batch = args.get('micro_batchsize', -1)
    if micro_batch == -1:
        micro_batch = args.batchsize

    try:
        for e in range(args.epoch, args.epochs):
            args.epoch = e
            dset = tqdm(iter(dataloader))
            for i, (seq, im) in enumerate(dset):
                if seq is not None and im is not None:
                    opt.zero_grad()
                    total_loss = 0
                    for j in range(0, len(im), micro_batch):
                        tgt_seq, tgt_mask = seq['input_ids'][j:j + micro_batch].to(device), seq['attention_mask'][
                                                                                           j:j + micro_batch].bool().to(
                            device)
                        loss = model.data_parallel(im[j:j + micro_batch].to(device), device_ids=args.gpu_devices,
                                                   t=tgt_seq, mask=tgt_mask) * micro_batch / args.batchsize
                        loss.backward()
                        total_loss += loss.item()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    opt.step()
                    scheduler.step()
                    dset.set_description(f'Loss: {total_loss:.4f}')
                    if args.wandb:
                        wandb.log({'train/loss': total_loss})
                if (i + 1 + len(dataloader) * e) % args.sample_freq == 0:
                    bleu_score, edit_distance, token_accuracy = evaluate(model, valdataloader, args, num_batches=int(
                        args.valbatches * e / args.epochs), name='val')
                    if bleu_score > max_bleu and token_accuracy > max_token_acc:
                        max_bleu, max_token_acc = bleu_score, token_accuracy
                        save_models(e, step=i)
            if (e + 1) % args.save_freq == 0:
                save_models(e, step=len(dataloader))
            if args.wandb:
                wandb.log({'train/epoch': e + 1})
    except KeyboardInterrupt:
        if e >= 2:
            save_models(e, step=i)
        raise KeyboardInterrupt
    save_models(e, step=len(dataloader))


@torch.no_grad()
def evaluate(model: Model, dataset: dataset.I2LDataset, args: Munch, num_batches: int = None, name: str = 'test'):
    def detokenize(tokens, tokenizer):
        toks = [tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
        for b in range(len(toks)):
            for i in reversed(range(len(toks[b]))):
                if toks[b][i] is None:
                    toks[b][i] = ''
                toks[b][i] = toks[b][i].replace('Ġ', ' ').strip()
                if toks[b][i] in ([BOS, EOS, PAD]):
                    del toks[b][i]
        return toks

    assert len(dataset) > 0
    bleus, edit_dists, token_acc, log = [], [], [], {}
    bleu_score, edit_distance, token_accuracy = 0, 1, 0
    pbar = tqdm(enumerate(iter(dataset)), total=len(dataset))
    for i, (seq, im) in pbar:
        if seq is None or im is None:
            continue
        dec = model.generate(im.to(args.device), temperature=args.get('temperature', .2))
        pred = detokenize(dec, dataset.tokenizer)
        truth = detokenize(seq['input_ids'], dataset.tokenizer)
        bleus.append(metrics.bleu_score(pred, [[x] for x in truth]))
        for predi, truthi in zip(tok2str(dec, dataset.tokenizer), tok2str(seq['input_ids'], dataset.tokenizer)):
            ts = post_process(truthi)
            if len(ts) > 0:
                edit_dists.append(Levenshtein.distance(post_process(predi), ts) / len(ts))
        dec = dec.cpu()
        tgt_seq = seq['input_ids'][:, 1:]
        shape_diff = dec.shape[1] - tgt_seq.shape[1]
        if shape_diff < 0:
            dec = torch.nn.functional.pad(dec, (0, -shape_diff), "constant", utils.PAD_ID)
        elif shape_diff > 0:
            tgt_seq = torch.nn.functional.pad(tgt_seq, (0, shape_diff), "constant", utils.PAD_ID)
        mask = torch.logical_or(tgt_seq != utils.PAD_ID, dec != utils.PAD_ID)
        tok_acc = (dec == tgt_seq)[mask].float().mean().item()
        token_acc.append(tok_acc)
        pbar.set_description(f'BLEU: {numpy.mean(bleus):.3f}, ED: {numpy.mean(edit_dists):.2e}, ACC: {numpy.mean(token_acc):.3f}')
        if num_batches is not None and i >= num_batches:
            break
    if len(bleus) > 0:
        bleu_score = numpy.mean(bleus)
        log[name + '/bleu'] = bleu_score
    if len(edit_dists) > 0:
        edit_distance = numpy.mean(edit_dists)
        log[name + '/edit_distance'] = edit_distance
    if len(token_acc) > 0:
        token_accuracy = numpy.mean(token_acc)
        log[name + '/token_acc'] = token_accuracy
    if args.wandb:
        pred = tok2str(dec, dataset.tokenizer)
        truth = tok2str(seq['input_ids'], dataset.tokenizer)
        table = wandb.Table(["Truth", "Prediction"])
        for k in range(min([len(pred), args.test_samples])):
            table.add_data(post_process(truth[k]), post_process(pred[k]))
        log[name + '/examples'] = table
        wandb.log(log)
    else:
        print(f'\n{truth}\n{pred}')
        print(f'BLEU: {bleu_score:.2f}')
    return bleu_score, edit_distance, token_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', default=None, help='path to yaml config file', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU')
    parser.add_argument('--debug', action='store_true', help='DEBUG')
    parser.add_argument('--resume', help='path to checkpoint folder', action='store_true')
    parsed_args = parser.parse_args()
    if not parsed_args.config:
        parsed_args.config = os.path.realpath('model/config.yaml')
    args = parse_args(Munch(yaml.load(open(parsed_args.config, 'r'), Loader=yaml.FullLoader)), **vars(parsed_args))
    logging.getLogger().setLevel(logging.DEBUG if parsed_args.debug else logging.WARNING)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if args.wandb:
        if not parsed_args.resume:
            args.id = wandb.util.generate_id()
        wandb.init(config=dict(args), resume='allow', name=args.name, id=args.id)
        args = Munch(wandb.config)
    train(args)
