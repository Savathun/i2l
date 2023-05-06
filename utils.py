import contextlib
import os
import re

import cv2
import numpy
import torch
from PIL import Image
import munch

CHANNELS = 1
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
PAD = "[PAD]"
BOS = "[BOS]"
EOS = "[EOS]"


def parse_args(args, **kwargs) -> munch.Munch:
    args = munch.Munch({'epoch': 0}, **args)

    def get_device(no_cuda=False):
        device = 'cpu'
        available_gpus = torch.cuda.device_count()
        args.gpu_devices = args.gpu_devices if args.get('gpu_devices', False) else list(range(available_gpus))
        if available_gpus > 0 and not no_cuda:
            device = f'cuda:{args.gpu_devices[0]:d}' if args.gpu_devices else 0
            assert available_gpus >= len(
                args.gpu_devices), f"Available {available_gpus:d} gpu, but specified gpu {','.join(map(str, args.gpu_devices))}."
            assert max(
                args.gpu_devices) < available_gpus, f"legal gpu_devices should in [{','.join(map(str, range(available_gpus)))}], received [{','.join(map(str, args.gpu_devices))}]"
        return device

    kwargs = munch.Munch({'no_cuda': False, 'debug': False}, **kwargs)
    args.update(kwargs)
    args.wandb = not kwargs.debug and not args.debug
    args.device = get_device(kwargs.no_cuda)
    args.max_dimensions = [args.max_width, args.max_height]
    args.min_dimensions = [args.get('min_width', 32), args.get('min_height', 32)]
    if 'decoder_args' not in args or not args.decoder_args:
        args.decoder_args = {}
    return args


def gpu_memory_check(model, args):
    try:
        batch_size = args.batchsize if args.get('micro_batchsize', -1) == -1 else args.micro_batchsize
        for _ in range(5):
            im = torch.empty(batch_size, CHANNELS, args.max_height, args.min_height, device=args.device).float()
            seq = torch.randint(0, args.num_tokens, (batch_size, args.max_seq_len), device=args.device).long()
            loss = model.data_parallel(im, device_ids=args.gpu_devices, tgt_seq=seq)
            loss.sum().backward()
    except RuntimeError:
        raise RuntimeError(
            f"The system cannot handle a batch size of {batch_size} for the maximum "
            f"image size ({args.max_height}, {args.max_width}). Try to use a smaller micro batchsize.")
    model.zero_grad()
    with torch.cuda.device(args.device):
        torch.cuda.empty_cache()
    del im, seq


def token2str(tokens, tokenizer) -> list:
    if len(tokens.shape) == 1:
        tokens = tokens[None, :]
    return [''.join(detok.split(' ')).replace('Ġ', ' ').replace(EOS, '').replace(BOS, '').replace(PAD,
                                                                                                  '').strip()
            for detok in [tokenizer.decode(tok) for tok in tokens]]


def pad(img: Image, divable: int = 32) -> Image:
    threshold = 128
    data = numpy.array(img.convert('LA'))
    if data[..., -1].var() == 0:
        data = (data[..., 0]).astype(numpy.uint8)
    else:
        data = (255 - data[..., -1]).astype(numpy.uint8)
    data = (data - data.min()) / (data.max() - data.min()) * 255
    if data.mean() > threshold:
        gray = 255 * (data < threshold).astype(numpy.uint8)
    else:
        gray = 255 * (data > threshold).astype(numpy.uint8)
        data = 255 - data
    a, b, w, h = cv2.boundingRect(cv2.findNonZero(gray))
    dims = []
    for x in [w, h]:
        div, mod = divmod(x, divable)
        dims.append(divable * (div + (1 if mod > 0 else 0)))
    padded = Image.new('L', dims, 255)
    im = Image.fromarray(data[b:b + h, a:a + w]).convert('L')
    padded.paste(im, (0, 0, im.size[0], im.size[1]))
    return padded


def post_process(s: str):
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter, no_letter = '[a-zA-Z]', r'[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(fr'(?!\\ )({no_letter})\s+?({no_letter})', r'\1\2', s)
        news = re.sub(fr'(?!\\ )({no_letter})\s+?({letter})', r'\1\2', news)
        news = re.sub(fr'({letter})\s+?({no_letter})', r'\1\2', news)
        if news == s:
            break
    return s


@contextlib.contextmanager
def in_model_path():
    model_path = os.path.join(os.path.dirname(__file__), 'model')
    saved = os.getcwd()
    os.chdir(model_path)
    try:
        yield
    finally:
        os.chdir(saved)
