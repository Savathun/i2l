import math
import re

import cv2
import numpy
import torch
from PIL import Image
import munch

CHANNELS = 1
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
PAD, BOS, EOS = "[PAD]", "[BOS]", "[EOS]"


def parse_args(args, **kwargs) -> munch.Munch:
    args = munch.Munch({'epoch': 0, 'decoder_args': {}, 'min_width': 32, 'min_height': 32, 'no_cuda': False,
                        'checkpoint': 'model/checkpoints/weights.pth'}, **args)

    def get_device() -> str:
        device = 'cpu'
        available_gpus = torch.cuda.device_count()
        args.gpu_devices = args.gpu_devices if args.get('gpu_devices', False) else list(range(available_gpus))
        if available_gpus > 0 and not args.no_cuda:
            device = f'cuda:{args.gpu_devices[0]:d}' if args.gpu_devices else 0
            assert available_gpus >= len(
                args.gpu_devices), f"Available {available_gpus:d} gpu, but specified gpu {','.join(map(str, args.gpu_devices))}."
            assert max(
                args.gpu_devices) < available_gpus, f"legal gpu_devices should in [{','.join(map(str, range(available_gpus)))}], received [{','.join(map(str, args.gpu_devices))}]"
        return device
    return munch.Munch(args, device=get_device(), max_dimensions=[args.max_width, args.max_height],
                       min_dimensions=[args.min_width, args.min_height], **kwargs)


def check_mem(model, args) -> None:
    try:
        batch_size = args.batchsize if args.get('micro_batchsize', -1) == -1 else args.micro_batchsize
        for _ in range(5):
            model.data_parallel(
                torch.empty(batch_size, CHANNELS, args.max_height, args.min_height, device=args.device).float(),
                args.gpu_devices, t=torch.randint(0, args.num_tokens, (batch_size, args.max_seq_len),
                                                  device=args.device).long()).sum().backward()
    except RuntimeError:
        raise RuntimeError(
            f"The system cannot handle a batch size of {batch_size} for the maximum "
            f"image size ({args.max_height}, {args.max_width}). Try to use a smaller micro batchsize.")
    model.zero_grad()
    with torch.cuda.device(args.device):
        torch.cuda.empty_cache()


def tok2str(tokens, tokenizer) -> list:
    if len(tokens.shape) == 1:
        tokens = tokens[None, :]
    return [re.sub(fr'(Ġ)|(\{PAD}|\{BOS}|\{EOS}| )', lambda m: ' ' if m.group(1) else '' if m.group(2) else None,
                   tokenizer.decode(tok)).strip() for tok in tokens]


def pad(img: Image, divisor=32) -> Image:
    if (data := numpy.array(img.convert('LA')))[..., -1].var() == 0:
        data = (data[..., 0]).astype(numpy.uint8)
    else:
        data = (255 - data[..., -1]).astype(numpy.uint8)
    data = (data - data.min()) / (data.max() - data.min()) * 255
    if data.mean() > (threshold := 128):
        gray = 255 * (data < threshold).astype(numpy.uint8)
    else:
        gray = 255 * (data > threshold).astype(numpy.uint8)
        data = 255 - data
    a, b, w, h = cv2.boundingRect(cv2.findNonZero(gray))
    padded = Image.new('L', [divisor * math.ceil(x/divisor) for x in [w, h]], 255)
    im = Image.fromarray(data[b:b + h, a:a + w]).convert('L')
    padded.paste(im, (0, 0, im.size[0], im.size[1]))
    return padded


def post_process(s: str) -> str:
    result = re.sub(r"(\\(?:operatorname|mathrm|text|mathbf))\s?(\*?)\s?({)\s?(.*?)\s?(})", r"\1\2\3\4\5", s)
    sub = lambda m: m.expand(r'\2\4') if m.group(2) and m.group(4) else m.expand(r'\3\5') if m.group(3) and m.group(5) else m.group(0)
    while (news := re.sub(r"(?!\\ )(([\W_^\d])|([a-zA-Z]))\s+?(([\W_^\d])|([a-zA-Z]))", sub, result)) != result:
        result = news
    return result

