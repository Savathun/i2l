import torch
import torch.nn as nn
import timm.models
from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper, autoregressive_wrapper
import einops
import utils

ENCODER_DEPTH = 4
HEADS = 8
ENCODER_LAYERS = 4


class Model(nn.Module):
    def __init__(self, encoder, decoder, args):
        super().__init__()
        self.encoder, self.decoder, self.args = encoder, decoder, args

    def data_parallel(self, x: torch.Tensor, device_ids, output_device=None, **kwargs):
        if not device_ids or len(device_ids) == 1:
            return self(x, **kwargs)
        if not output_device:
            output_device = device_ids[0]
        inputs = nn.parallel.scatter(x, device_ids)
        return nn.parallel.gather(
            nn.parallel.parallel_apply(nn.parallel.replicate(self, device_ids)[:len(inputs)], inputs,
                                       nn.parallel.scatter(kwargs, device_ids)[:len(inputs)]), output_device).mean()

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        return self.decoder(t, context=self.encoder(x), **kwargs)

    @torch.no_grad()
    def generate(self, x: torch.Tensor, temperature: float = 0.25):
        return self.decoder.generate((torch.LongTensor([utils.BOS_ID] * len(x))[:, None]).to(x.device),
                                     self.args.max_seq_len,
                                     context=self.encoder(x), temperature=temperature)


def get_model(args):
    class ARWrapperWithCustomGenerate(AutoregressiveWrapper):
        def __init__(self, *args, **kwargs):
            super(ARWrapperWithCustomGenerate, self).__init__(*args, **kwargs)

        @torch.no_grad()
        def generate(self, start_tokens, seq_len=256, temperature=1., filter_logits_fn=autoregressive_wrapper.top_k,
                     filter_thres=0.9, **kwargs):
            num_dims = len(start_tokens.shape)

            if num_dims == 1:
                start_tokens = start_tokens[None, :]

            b, t = start_tokens.shape

            self.net.eval()
            out = start_tokens
            mask = kwargs.pop('mask', None)
            if not mask:
                mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

            for _ in range(seq_len):
                x = out[:, -self.max_seq_len:]
                mask = mask[:, -self.max_seq_len:]
                logits = self.net(x, mask=mask, **kwargs)[:, -1, :]

                if filter_logits_fn in {autoregressive_wrapper.top_k, autoregressive_wrapper.top_p}:
                    filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                    probs = nn.functional.softmax(filtered_logits / temperature, dim=-1)

                out = torch.cat((out, torch.multinomial(probs, 1)), dim=-1)
                mask = nn.functional.pad(mask, (0, 1), value=True)

                if utils.EOS_ID and (torch.cumsum(out == utils.EOS_ID, 1)[:, -1] >= 1).all():
                    break

            out = out[:, t:]

            if num_dims == 1:
                out = out.squeeze(0)

            self.net.train(self.net.training)
            return out

    decoder = ARWrapperWithCustomGenerate(TransformerWrapper(num_tokens=args.num_tokens, max_seq_len=args.max_seq_len,
                                                 attn_layers=Decoder(dim=args.dim, depth=ENCODER_LAYERS,
                                                                     heads=HEADS, **args.decoder_args)),
                                          pad_value=utils.PAD_ID)
    encoder = get_encoder(args)
    encoder.to(args.device)
    decoder.to(args.device)
    model = Model(encoder, decoder, args)
    if args.wandb:
        import wandb
        wandb.watch(model)

    return model


def get_encoder(args):
    class ViTWithCustomForward(timm.models.vision_transformer.VisionTransformer):
        def __init__(self, img_size=224, patch_size=16, *args, **kwargs):
            super(ViTWithCustomForward, self).__init__(img_size=img_size, patch_size=patch_size, *args, **kwargs)
            self.height, self.width = img_size
            self.patch_size = patch_size

        def forward_features(self, x):
            B, c, h, w = x.shape
            h, w = h // self.patch_size, w // self.patch_size
            x = self.pos_drop(torch.cat((self.cls_token.expand(B, -1, -1), self.patch_embed(x)), dim=1) +
                              self.pos_embed[:,
                              torch.cat((torch.zeros(1), einops.repeat(torch.arange(h) * (self.width // self.patch_size - w),
                                                                'h -> (h w)', w=w) + torch.arange(h * w) + 1),
                                        dim=0).long()])
            for blk in self.blocks:
                x = blk(x)
            return self.norm(x)

    backbone = timm.models.resnetv2.ResNetV2(layers=args.backbone_layers, num_classes=0, global_pool='',
                                             in_chans=utils.CHANNELS, preact=False, stem_type='same',
                                             conv_layer=timm.models.layers.StdConv2dSame)
    min_patch_size = 2 ** (len(args.backbone_layers) + 1)

    def embed_layer(**x):
        ps = x.pop('patch_size', min_patch_size)
        assert ps % min_patch_size == 0 and ps >= min_patch_size, f'patch_size needs to be multiple of {min_patch_size}'
        return timm.models.vision_transformer_hybrid.HybridEmbed(**x, patch_size=ps // min_patch_size,
                                                                 backbone=backbone)

    encoder = ViTWithCustomForward(img_size=(args.max_height, args.max_width), patch_size=args.patch_size,
                                      in_chans=utils.CHANNELS, num_classes=0, embed_dim=args.dim,
                                      depth=ENCODER_DEPTH, num_heads=HEADS, embed_layer=embed_layer)
    return encoder
