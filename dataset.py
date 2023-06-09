import glob
import logging
import os
import pickle
import collections
from os.path import join
import argparse
import albumentations as alb
import albumentations.pytorch

import cv2
import imagesize
import numpy
import torch
from torch.nn.utils.rnn import pad_sequence
import tqdm.auto
import transformers
import tokenizers
from utils import PAD_ID, BOS_ID, EOS_ID, PAD, BOS, EOS

train_transform = alb.Compose([alb.Compose([alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.15, 0), rotate_limit=1,
                                                                 border_mode=0, interpolation=3, value=[255, 255, 255],
                                                                 p=1),
                                            alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                                                               value=[255, 255, 255], p=.5)], p=.15),
                               alb.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                               alb.GaussNoise(10, p=.2),
                               alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
                               alb.ImageCompression(95, p=.3),
                               alb.ToGray(always_apply=True),
                               alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
                               albumentations.pytorch.ToTensorV2()])
test_transform = alb.Compose([alb.ToGray(always_apply=True),
                              alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
                              albumentations.pytorch.ToTensorV2()])


def load(f: str): return pickle.load(open(f, 'rb'))


class I2LDataset:
    keep_smaller_batches = False
    shuffle = True
    batchsize = 16
    max_dimensions = (1024, 512)
    min_dimensions = (32, 32)
    max_seq_len = 1024

    transform = train_transform
    data = collections.defaultdict(lambda: [])

    def __init__(self, equations=None, images=None, tokenizer=None, shuffle=True, batchsize=16, max_seq_len=1024,
                 max_dimensions=(1024, 512), min_dimensions=(32, 32), pad=False, keep_smaller_batches=False,
                 test=False):
        if images and equations:
            assert tokenizer
            self.images = [path.replace('\\', '/') for path in glob.glob(join(images, '*.png'))]
            self.sample_size = len(self.images)
            self.indices = [int(os.path.basename(img).split('.')[0]) for img in self.images]
            self.tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file=tokenizer)
            self.batchsize, self.shuffle, self.max_seq_len, self.pad = batchsize, shuffle, max_seq_len, pad
            self.max_dimensions, self.min_dimensions, self.test = max_dimensions, min_dimensions, test
            self.keep_smaller_batches = keep_smaller_batches
            try:
                eqs = open(equations, 'r').read().split('\n')
                for i, im in tqdm.auto.tqdm(enumerate(self.images), total=len(self.images)):
                    width, height = imagesize.get(im)
                    if min_dimensions[0] <= width <= max_dimensions[0] and \
                            min_dimensions[1] <= height <= max_dimensions[1]:
                        self.data[(width, height)].append((eqs[self.indices[i]], im))
            except KeyboardInterrupt: pass
            self.data = dict(self.data)
            self._get_size()
            iter(self)

    def __len__(self): return self.size

    def __iter__(self):
        self.i = 0
        self.transform = test_transform if self.test else train_transform
        self.pairs = []
        for k in self.data:
            info = numpy.array(self.data[k], dtype=object)
            p = torch.randperm(len(info)) if self.shuffle else torch.arange(len(info))
            for i in range(0, len(info), self.batchsize):
                batch = info[p[i:i + self.batchsize]]
                if len(batch.shape) == 1:
                    batch = batch[None, :]
                if len(batch) < self.batchsize and not self.keep_smaller_batches:
                    continue
                self.pairs.append(batch)
        self.pairs = numpy.random.permutation(numpy.array(
            self.pairs, object)) if self.shuffle else numpy.array(self.pairs, object)
        self.size = len(self.pairs)
        return self

    def __next__(self):
        if self.i >= self.size:
            raise StopIteration
        self.i += 1
        eqs, ims = self.pairs[self.i - 1].T
        tok = self.tokenizer(list(eqs), return_token_type_ids=False)
        for k, p in zip(tok, [[BOS_ID, EOS_ID], [1, 1]]):
            tok[k] = pad_sequence([torch.LongTensor([p[0]] + x + [p[1]]) for x in tok[k]], True, PAD_ID)
        if self.max_seq_len < tok['attention_mask'].shape[1]:
            return next(self)
        images = []
        for path in list(ims):
            if (im := cv2.imread(path)) is None:
                print(path, 'not found!')
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if not self.test:
                if numpy.random.random() < .04:
                    im[im != 255] = 0
            images.append(self.transform(image=im)['image'][:1])
        try:
            images = torch.cat(images).float().unsqueeze(1)
        except RuntimeError:
            logging.critical('Images not working: ' ' '.join(list(ims)))
            return None, None
        if self.pad:
            images = torch.nn.functional.pad(images, value=1, pad=(0, self.max_dimensions[0] - images.shape[2:][1], 0,
                                                                   self.max_dimensions[1] - images.shape[2:][0]))
        return tok, images

    def _get_size(self): self.size = sum([len(self.data[k]) // self.batchsize for k in self.data])

    def save(self, filename: str): pickle.dump(self, open(filename, 'wb'))

    def update(self, **kwargs):
        for k in ['batchsize', 'shuffle', 'pad', 'keep_smaller_batches', 'test', 'max_seq_len']:
            if k in kwargs:
                setattr(self, k, kwargs[k])
        if 'max_dimensions' in kwargs or 'min_dimensions' in kwargs:
            if 'max_dimensions' in kwargs:
                self.max_dimensions = kwargs['max_dimensions']
            if 'min_dimensions' in kwargs:
                self.min_dimensions = kwargs['min_dimensions']
            self.data = {k: self.data[k] for k in self.data if self.min_dimensions[0] <= k[0] <= self.max_dimensions[0] and
                         self.min_dimensions[1] <= k[1] <= self.max_dimensions[1]}
        if 'tokenizer' in kwargs:
            self.tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file=kwargs['tokenizer'])
        self._get_size()
        iter(self)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model', add_help=False)
    parser.add_argument('-i', '--images', type=str, default=None, help='image folder')
    parser.add_argument('-e', '--equations', type=str, default=r'C:\Users\djsch\Documents\GitHub\i2l\model\data\formulae.txt', help='equations text file')
    parser.add_argument('-t', '--tokenizer', default='model/tokenizer.json', help='pretrained tokenizer file')
    parser.add_argument('-o', '--out', type=str, help='output file')
    parser.add_argument('-s', '--vocab-size', default=8000, type=int, help='vocabulary size when training a tokenizer')
    args = parser.parse_args()
    if not args.images:
        if not args.out: args.out = args.tokenizer
        print('Generate tokenizer')
        tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.train([args.equations], tokenizers.trainers.BpeTrainer(special_tokens=[PAD, BOS, EOS],
                                                                         vocab_size=args.vocab_size, show_progress=True))
        tokenizer.save(args.out)
    elif args.images:
        if not args.out: args.out = args.images + ".pkl"
        print('Generate dataset pickles')
        dataset = I2LDataset(args.equations, args.images, args.tokenizer)
        dataset.update(batchsize=1, keep_smaller_batches=True)
        dataset.save(args.out)
