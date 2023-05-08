import logging

import munch
import numpy
import torch
import transformers
import yaml
from PIL import Image
from timm.models.layers import StdConv2dSame
from timm.models.resnetv2 import ResNetV2

import dataset
import model
import utils


class Image2Latex:
    def __init__(self):
        logging.getLogger().setLevel(logging.FATAL)
        self.args = utils.parse_args(munch.Munch(yaml.load(open('model/config.yaml', 'r'), yaml.FullLoader)),
                                     no_cuda=True, wandb=False)
        self.model = model.get_model(self.args)
        self.model.load_state_dict(torch.load(self.args.checkpoint, self.args.device))
        self.model.eval()
        self.image_resizer = ResNetV2(layers=[2, 3, 3], num_classes=max(self.args.max_dimensions) // 32, in_chans=utils.CHANNELS,
                                      drop_rate=.05, stem_type='same', conv_layer=StdConv2dSame).to(self.args.device)
        self.image_resizer.load_state_dict(torch.load('model/checkpoints/image_resizer.pth', self.args.device))
        self.image_resizer.eval()
        self.tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file=self.args.tokenizer)

    @torch.no_grad()
    def predict(self, img) -> str:
        img = self.minmax_size(utils.pad(img))
        input_image = img.convert('RGB').copy()
        r, w, h = 1, input_image.size[0], input_image.size[1]
        for _ in range(10):
            h = int(h * r)
            img = utils.pad(self.minmax_size(input_image.resize(
                (w, h), Image.Resampling.BILINEAR if r > 1 else Image.Resampling.LANCZOS)))
            t = dataset.test_transform(image=numpy.array(img.convert('RGB')))['image'][:1].unsqueeze(0)
            w = 32 * (1 + self.image_resizer(t.to(self.args.device)).argmax(-1).item())
            if w == img.size[0]:
                break
            r = w / img.size[0]
        return utils.post_process(utils.tok2str(self.model.generate(
            t.to(self.args.device).to(self.args.device), self.args.get('temperature', .25)), self.tokenizer)[0])

    def minmax_size(self, img: Image) -> Image:
        if self.args.max_dimensions:
            ratios = [a / b for a, b in zip(img.size, self.args.max_dimensions)]
            if any([r > 1 for r in ratios]):
                img = img.resize((numpy.array(img.size) // max(ratios)).astype(int), Image.BILINEAR)
        if self.args.min_dimensions and (
                padded_size := [max(i_dim, m_dim) for i_dim, m_dim in zip(img.size, self.args.min_dimensions)]) != list(img.size):
            (padded_im := Image.new('L', padded_size, 255)).paste(img, img.getbbox())
            img = padded_im
        return img
