import logging
import os

import yaml
import torch
import numpy
from PIL import Image

import dataset

import transformers
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame

import model
import utils
import munch


class Image2Latex:

    image_resizer = None
    last_pic = None

    @utils.in_model_path()
    def __init__(self, arguments=None):
        if not arguments:
            arguments = munch.Munch(
                {'config': 'settings/config.yaml', 'checkpoint': 'checkpoints/weights.pth', 'no_cuda': True,
                 'no_resize': False})
        logging.getLogger().setLevel(logging.FATAL)
        params = yaml.load(open(arguments.config, 'r'), Loader=yaml.FullLoader)
        self.args = utils.parse_args(munch.Munch(params))
        self.args.update(**vars(arguments))
        self.args.wandb = False
        self.args.device = 'cuda' if torch.cuda.is_available() and not self.args.no_cuda else 'cpu'
        self.model = model.get_model(self.args)
        self.model.load_state_dict(torch.load(self.args.checkpoint, map_location=self.args.device))
        self.model.eval()

        if 'image_resizer.pth' in os.listdir(os.path.dirname(self.args.checkpoint)) and not arguments.no_resize:
            self.image_resizer = ResNetV2(layers=[2, 3, 3], num_classes=max(self.args.max_dimensions) // 32,
                                          global_pool='avg', in_chans=1, drop_rate=.05,
                                          preact=True, stem_type='same', conv_layer=StdConv2dSame).to(self.args.device)
            self.image_resizer.load_state_dict(
                torch.load(os.path.join(os.path.dirname(self.args.checkpoint), 'image_resizer.pth'),
                           map_location=self.args.device))
            self.image_resizer.eval()
        self.tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file=self.args.tokenizer)

    @utils.in_model_path()
    def predict(self, img=None, resize=True) -> str:
        if type(img) is bool:
            img = None
        if not img:
            if not self.last_pic:
                return ''
            else:
                print('\nLast image is: ', end='')
                img = self.last_pic.copy()
        else:
            self.last_pic = img.copy()
        img = minmax_size(utils.pad(img), self.args.max_dimensions, self.args.min_dimensions)
        if (self.image_resizer and not self.args.no_resize) and resize:
            with torch.no_grad():
                input_image = img.convert('RGB').copy()
                r, w, h = 1, input_image.size[0], input_image.size[1]
                for _ in range(10):
                    h = int(h * r)  # height to resize
                    img = utils.pad(minmax_size(
                        input_image.resize((w, h), Image.Resampling.BILINEAR if r > 1 else Image.Resampling.LANCZOS),
                        self.args.max_dimensions, self.args.min_dimensions))
                    t = dataset.test_transform(image=numpy.array(img.convert('RGB')))['image'][:1].unsqueeze(0)
                    w = (self.image_resizer(t.to(self.args.device)).argmax(-1).item() + 1) * 32
                    logging.info(r, img.size, (w, int(input_image.size[1] * r)))
                    if w == img.size[0]:
                        break
                    r = w / img.size[0]
        else:
            img = numpy.array(img.convert('RGB'))
            t = dataset.test_transform(image=img)['image'][:1].unsqueeze(0)
        im = t.to(self.args.device)

        dec = self.model.generate(im.to(self.args.device), temperature=self.args.get('temperature', .25))
        pred = utils.post_process(utils.token2str(dec, self.tokenizer)[0])
        return pred


def minmax_size(img: Image, max_dimensions: tuple[int, int] = None, min_dimensions: tuple[int, int] = None) -> Image:
    if max_dimensions:
        ratios = [a / b for a, b in zip(img.size, max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = numpy.array(img.size) // max(ratios)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions:
        padded_size = [max(img_dim, min_dim) for img_dim, min_dim in zip(img.size, min_dimensions)]
        if padded_size != list(img.size):  # assert hypothesis
            padded_im = Image.new('L', padded_size, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img
