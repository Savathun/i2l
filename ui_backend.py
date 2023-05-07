import logging
import os

import yaml
import torch
import numpy
from PIL import Image
import munch
import transformers
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame

import model
import utils
import dataset


class Image2Latex:
    @utils.in_model_path()
    def __init__(self, arguments=None):
        if not arguments:
            arguments = munch.Munch(
                {'config': 'settings/config.yaml', 'checkpoint': 'checkpoints/weights.pth', 'no_cuda': True})
        logging.getLogger().setLevel(logging.FATAL)
        params = yaml.load(open(arguments.config, 'r'), yaml.FullLoader)
        self.args = utils.parse_args(munch.Munch(params))
        self.args.update(**vars(arguments))
        self.args.device = 'cuda' if torch.cuda.is_available() and not self.args.no_cuda else 'cpu'
        self.args.wandb = False
        self.model = model.get_model(self.args)
        self.model.load_state_dict(torch.load(self.args.checkpoint, self.args.device))
        self.model.eval()
        self.image_resizer = ResNetV2(layers=[2, 3, 3], num_classes=max(self.args.max_dimensions) // 32, in_chans=1,
                                      drop_rate=.05, stem_type='same', conv_layer=StdConv2dSame).to(self.args.device)
        self.image_resizer.load_state_dict(
            torch.load(os.path.join(os.path.dirname(self.args.checkpoint), 'image_resizer.pth'), self.args.device))
        self.image_resizer.eval()
        self.tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file=self.args.tokenizer)

    @utils.in_model_path()
    def predict(self, img) -> str:
        img = minmax_size(utils.pad(img), self.args.max_dimensions, self.args.min_dimensions)
        with torch.no_grad():
            input_image = img.convert('RGB').copy()
            r, w, h = 1, input_image.size[0], input_image.size[1]
            for _ in range(10):
                h = int(h * r)
                img = utils.pad(minmax_size(
                    input_image.resize((w, h), Image.Resampling.BILINEAR if r > 1 else Image.Resampling.LANCZOS),
                    self.args.max_dimensions, self.args.min_dimensions))
                t = dataset.test_transform(image=numpy.array(img.convert('RGB')))['image'][:1].unsqueeze(0)
                w = (self.image_resizer(t.to(self.args.device)).argmax(-1).item() + 1) * 32
                logging.info(r, img.size, (w, int(input_image.size[1] * r)))
                if w == img.size[0]:
                    break
                r = w / img.size[0]
        return utils.post_process(utils.token2str(self.model.generate(
            t.to(self.args.device).to(self.args.device), self.args.get('temperature', .25)), self.tokenizer)[0])


def minmax_size(img: Image, max_dims: tuple[int, int] = None, min_dims: tuple[int, int] = None) -> Image:
    if max_dims:
        ratios = [a / b for a, b in zip(img.size, max_dims)]
        if any([r > 1 for r in ratios]):
            img = img.resize((numpy.array(img.size) // max(ratios)).astype(int), Image.BILINEAR)
    if min_dims and (padded_size := [max(idim, mdim) for idim, mdim in zip(img.size, min_dims)]) != list(img.size):
        (padded_im := Image.new('L', padded_size, 255)).paste(img, img.getbbox())
        img = padded_im
    return img
