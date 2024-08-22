# from vit_model import ViTModel
# from base_model import BaseModel, BaseMixin
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import mindspore as ms
from mindspore import ops, nn
from mindone.transformers.models.blip_2.modeling_blip_2 import Blip2QFormerModel
from mindone.transformers.models.clip.modeling_clip import CLIPVisionTransformer


def non_conflict(func):
    '''mark a hook function as non-conflict,
    so that it can be compatible with any already defined hooks.
    e.g. PrefixTuningMixin.attention_fn
    '''
    func.non_conflict = True
    return func


def replacable(func):
    '''mark a hook function as replacable,
    so that it can be replaced by mixins added after it.
    e.g. FP32AttentionMixin.attention_fn
    '''
    func.replacable = True
    return func


class BaseMixin(nn.Cell):
    non_conflict = non_conflict
    replacable = replacable

    def __init__(self):
        super(BaseMixin, self).__init__()
        # define new params

    def reinit(self, parent_model=None):
        # reload the initial params from previous trained modules
        # you can also get access to other mixins through parent_model.get_mixin().
        pass


class LNFinalyMixin(BaseMixin):
    def __init__(self, hidden_size):
        super().__init__()
        self.ln_vision = nn.LayerNorm(hidden_size)

    def final_forward(self, logits, **kw_args):
        return self.ln_vision(logits)


# class EVAViT(ViTModel):
#     def __init__(self, args, transformer=None, parallel_output=True, **kwargs):
#         super().__init__(args, transformer=transformer, parallel_output=parallel_output, **kwargs)
#         self.del_mixin("cls")
#         self.add_mixin("cls", LNFinalyMixin(args.hidden_size))
#
#     def forward(self, image):
#         batch_size = image.size(0)
#         input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=image.device)
#         attention_mask = torch.tensor([[1.]], dtype=image.dtype, device=image.device)
#         return super().forward(input_ids=input_ids, position_ids=None, attention_mask=attention_mask, image=image)


# class QFormer(BaseModel):
#     def __init__(self, args, transformer=None, parallel_output=True, **kwargs):
#         super().__init__(args, transformer=transformer, parallel_output=parallel_output,
#                          activation_func=nn.functional.gelu, **kwargs)
#         self.transformer.position_embeddings = None
#
#     def final_forward(self, logits, **kw_args):
#         return logits
#
#     def position_embedding_forward(self, position_ids, **kw_args):
#         return None
#
#     def forward(self, encoder_outputs):
#         batch_size = encoder_outputs.size(0)
#         input_ids = torch.arange(32, dtype=torch.long, device=encoder_outputs.device).unsqueeze(0).expand(batch_size,
#                                                                                                           -1)
#         attention_mask = torch.tensor([[1.]], dtype=encoder_outputs.dtype, device=encoder_outputs.device)
#         cross_attention_mask = torch.tensor([[1.]], dtype=encoder_outputs.dtype, device=encoder_outputs.device)
#         return super().forward(input_ids=input_ids, position_ids=None, attention_mask=attention_mask,
#                                encoder_outputs=encoder_outputs, cross_attention_mask=cross_attention_mask)


class BLIP2(nn.Cell):
    def __init__(self, eva_args, qformer_args, vit=None, qformer=None, **kwargs):
        super().__init__()
        if vit is not None:
            self.vit = vit
        else:
            self.vit = CLIPVisionTransformer(CLIPVisionTransformer.get_args(**eva_args))
        if qformer is not None:
            self.qformer = qformer
        else:
            self.qformer = Blip2QFormerModel(Blip2QFormerModel.get_args(**qformer_args))

        self.glm_proj = nn.Dense(768, 4096)

    def forward(self, image, **kwargs):
        enc = self.vit(image)[0]
        out = self.qformer(enc)[0]
        return self.glm_proj(out)


class BlipImageBaseProcessor():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)


class BlipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)
