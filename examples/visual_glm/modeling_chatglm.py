""" PyTorch ChatGLM model. """

import inspect
import math
import copy
import warnings
import re
import sys
import requests

# from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

import mindspore as ms
from mindspore import nn, ops

from mindone.transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithPast
)
from transformers.utils import logging
from transformers.generation.utils import ModelOutput

from .configuration_chatglm import ChatGLMConfig

from mindone.transformers import MSPreTrainedModel
from utils import LogitsProcessorList

# flags required to enable jit fusion kernels

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "THUDM/ChatGLM-6B"
_CONFIG_FOR_DOC = "ChatGLM6BConfig"

CHATGLM_6B_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "THUDM/chatglm-6b",
    # See all ChatGLM-6B models at https://huggingface.co/models?filter=chatglm
]

class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(self, input_ids: ms.Tensor, scores: ms.Tensor) -> ms.Tensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        if ops.isnan(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class ImagePatchEmbedding(nn.Cell):
    def __init__(self, in_channels, hidden_size, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def construct(self, images):
        """
        Input:
        * images with shape (B, C, H, W)
        Output:
        * (batch_size, hidden_size)
        """
        embeddings = self.proj(images)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        return embeddings


class PrefixEncoder(nn.Cell):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = nn.SequentialCell(
                nn.Dense(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Dense(config.hidden_size, config.num_layers * config.hidden_size * 2)
            )
        else:
            self.embedding = nn.Embedding(config.pre_seq_len, config.num_layers * config.hidden_size * 2)

    def forward(self, prefix: ms.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


# @torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + ops.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


def gelu(x):
    return gelu_impl(x)


class RotaryEmbedding(nn.Cell):
    def __init__(self, dim, base=10000, precision=ms.half, learnable=False):
        super().__init__()
        inv_freq = 1. / (base ** (ops.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq.half()
        self.learnable = learnable
        if learnable:
            self.inv_freq = ms.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        pass

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = ops.arange(seq_len, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # 向量外积
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = ops.cat((freqs, freqs), axis=-1).to(x.device)
            if self.precision == ms.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == ms.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def _apply(self, fn):
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn)


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return ops.cat((-x2, x1), axis=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


# @torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
        F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k


def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        hidden_size_per_partition,
        layer_id,
        layer_past=None,
        scaling_attention_score=True,
        use_cache=False,
):
    if layer_past is not None:
        past_key, past_value = layer_past[0], layer_past[1]
        key_layer = ops.cat((past_key, key_layer), axis=0)
        value_layer = ops.cat((past_value, value_layer), axis=0)

    # seqlen, batch, num_attention_heads, hidden_size_per_attention_head
    seq_len, b, nh, hidden_size = key_layer.shape

    if use_cache:
        present = (key_layer, value_layer)
    else:
        present = None

    query_key_layer_scaling_coeff = float(layer_id + 1)
    if scaling_attention_score:
        query_layer = query_layer / (math.sqrt(hidden_size) * query_key_layer_scaling_coeff)

    # ===================================
    # Raw attention scores. [b, np, s, s]
    # ===================================

    # [b, np, sq, sk]
    output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

    # [sq, b, np, hn] -> [sq, b * np, hn]
    query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
    # [sk, b, np, hn] -> [sk, b * np, hn]
    key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

    matmul_result = ops.zeros(
        (1, 1, 1),
        dtype=query_layer.dtype
    )

    matmul_result = ops.baddbmm(
        matmul_result,
        query_layer.transpose(0, 1),  # [b * np, sq, hn]
        key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        beta=0.0,
        alpha=1.0,
    )

    # change view to [b, np, sq, sk]
    attention_scores = matmul_result.view(*output_size)

    if self.scale_mask_softmax:
        self.scale_mask_softmax.scale = query_key_layer_scaling_coeff
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask.contiguous())
    else:
        if not (attention_mask == 0).all():
            # if auto-regressive, skip
            attention_scores.masked_fill_(attention_mask, -10000.0)
        dtype = attention_scores.dtype
        attention_scores = attention_scores.float()
        attention_scores = attention_scores * query_key_layer_scaling_coeff

        attention_probs = ops.softmax(attention_scores, axis=-1)

        attention_probs = attention_probs.type(dtype)

    # =========================
    # Context layer. [sq, b, hp]
    # =========================

    # value_layer -> context layer.
    # [sk, b, np, hn] --> [b, np, sq, hn]

    # context layer shape: [b, np, sq, hn]
    output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

    # change view [sk, b * np, hn]
    value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

    # change view [b * np, sq, sk]
    attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

    # matmul: [b * np, sq, hn]
    context_layer = ops.bmm(attention_probs, value_layer.transpose(0, 1))

    # change view [b, np, sq, hn]
    context_layer = context_layer.view(*output_size)

    # [b, np, sq, hn] --> [sq, b, np, hn]
    context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

    # [sq, b, np, hn] --> [sq, b, hp]
    new_context_layer_shape = context_layer.size()[:-2] + (hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer, present, attention_probs)

    return outputs


def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)


class SelfAttention(nn.Cell):
    def __init__(self, hidden_size, num_attention_heads,
                 layer_id, hidden_size_per_attention_head=None, bias=True,
                 params_dtype=ms.float32, position_encoding_2d=True, empty_init=True):
        init_method = default_init
        super(SelfAttention, self).__init__()

        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.hidden_size_per_partition = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_heads_per_partition = num_attention_heads
        self.position_encoding_2d = position_encoding_2d
        self.rotary_emb = RotaryEmbedding(
            self.hidden_size // (self.num_attention_heads * 2)
            if position_encoding_2d
            else self.hidden_size // self.num_attention_heads,
            base=10000,
            precision=ms.half,
            learnable=False,
        )

        self.scale_mask_softmax = None

        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head

        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head

        # Strided linear layer.
        self.query_key_value = init_method(
            nn.Dense,
            hidden_size,
            3 * self.inner_hidden_size,
            bias=bias,
            dtype=params_dtype,
        )

        self.dense = init_method(
            nn.Dense,
            self.inner_hidden_size,
            hidden_size,
            bias=bias,
            dtype=params_dtype,
        )

    @staticmethod
    def attention_mask_func(attention_scores, attention_mask):
        attention_scores.masked_fill_(attention_mask, -10000.0)
        return attention_scores

    def split_tensor_along_last_dim(self, tensor, num_partitions,
                                    contiguous_split_chunks=False):
        """Split a tensor along its last dimension.
        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                    in memory.
        """
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = ops.split(tensor, last_dim_size, axis=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list

    def forward(
            self,
            hidden_states: ms.Tensor,
            position_ids,
            attention_mask: ms.Tensor,
            layer_id,
            layer_past: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ):
        """
        hidden_states: [seq_len, batch, hidden_size]
        attention_mask: [(1, 1), seq_len, seq_len]
        """

        # [seq_len, batch, 3 * hidden_size]
        mixed_raw_layer = self.query_key_value(hidden_states)

        # [seq_len, batch, 3 * hidden_size] --> [seq_len, batch, num_attention_heads, 3 * hidden_size_per_attention_head]
        new_tensor_shape = mixed_raw_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)

        # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
        (query_layer, key_layer, value_layer) = self.split_tensor_along_last_dim(mixed_raw_layer, 3)

        if self.position_encoding_2d:
            q1, q2 = query_layer.chunk(2, axis=(query_layer.ndim - 1))
            k1, k2 = key_layer.chunk(2, axis=(key_layer.ndim - 1))
            cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
            position_ids, block_position_ids = position_ids[:, 0, :].transpose(0, 1).contiguous(), \
                position_ids[:, 1, :].transpose(0, 1).contiguous()
            q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
            q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
            query_layer = ops.concat([q1, q2], axis=(q1.ndim - 1))
            key_layer = ops.concat([k1, k2], axis=(k1.ndim - 1))
        else:
            position_ids = position_ids.transpose(0, 1)
            cos, sin = self.rotary_emb(value_layer, seq_len=position_ids.max() + 1)
            # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
            query_layer, key_layer = apply_rotary_pos_emb_index(query_layer, key_layer, cos, sin, position_ids)

        # [seq_len, batch, hidden_size]
        context_layer, present, attention_probs = attention_fn(
            self=self,
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            attention_mask=attention_mask,
            hidden_size_per_partition=self.hidden_size_per_partition,
            layer_id=layer_id,
            layer_past=layer_past,
            use_cache=use_cache
        )

        output = self.dense(context_layer)

        outputs = (output, present)

        if output_attentions:
            outputs += (attention_probs,)

        return outputs  # output, present, attention_probs


class GEGLU(nn.Cell):
    def __init__(self):
        super().__init__()
        self.activation_fn = ops.gelu

    def forward(self, x):
        # dim=-1 breaks in jit for pt<1.10
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))
        return x1 * self.activation_fn(x2)


class GLU(nn.Cell):
    def __init__(self, hidden_size, inner_hidden_size=None,
                 layer_id=None, bias=True, activation_func=gelu, params_dtype=ms.float32, empty_init=True):
        super(GLU, self).__init__()
        init_method = default_init
        self.layer_id = layer_id
        self.activation_func = activation_func

        # Project to 4h.
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.dense_h_to_4h = init_method(
            nn.Dense,
            self.hidden_size,
            self.inner_hidden_size,
            bias=bias,
            dtype=params_dtype,
        )
        # Project back to h.
        self.dense_4h_to_h = init_method(
            nn.Dense,
            self.inner_hidden_size,
            self.hidden_size,
            bias=bias,
            dtype=params_dtype,
        )

    def forward(self, hidden_states):
        """
        hidden_states: [seq_len, batch, hidden_size]
        """

        # [seq_len, batch, inner_hidden_size]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)

        intermediate_parallel = self.activation_func(intermediate_parallel)

        output = self.dense_4h_to_h(intermediate_parallel)

        return output


class GLMBlock(nn.Dense):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            layernorm_epsilon,
            layer_id,
            inner_hidden_size=None,
            hidden_size_per_attention_head=None,
            layernorm=nn.LayerNorm,
            use_bias=True,
            params_dtype=ms.float32,
            num_layers=28,
            position_encoding_2d=True,
            empty_init=True
    ):
        super(GLMBlock, self).__init__()
        # Set output layer initialization if not provided.

        self.layer_id = layer_id

        # Layernorm on the input data.
        self.input_layernorm = layernorm(hidden_size, epsilon=layernorm_epsilon)

        self.position_encoding_2d = position_encoding_2d

        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            layer_id,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            bias=use_bias,
            params_dtype=params_dtype,
            position_encoding_2d=self.position_encoding_2d,
            empty_init=empty_init
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = layernorm(hidden_size, epsilon=layernorm_epsilon)

        self.num_layers = num_layers

        # GLU
        self.mlp = GLU(
            hidden_size,
            inner_hidden_size=inner_hidden_size,
            bias=use_bias,
            layer_id=layer_id,
            params_dtype=params_dtype,
            empty_init=empty_init
        )

    def forward(
            self,
            hidden_states: ms.Tensor,
            position_ids,
            attention_mask: ms.Tensor,
            layer_id,
            layer_past: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ):
        """
        hidden_states: [seq_len, batch, hidden_size]
        attention_mask: [(1, 1), seq_len, seq_len]
        """

        # Layer norm at the begining of the transformer layer.
        # [seq_len, batch, hidden_size]
        attention_input = self.input_layernorm(hidden_states)

        # Self attention.
        attention_outputs = self.attention(
            attention_input,
            position_ids,
            attention_mask=attention_mask,
            layer_id=layer_id,
            layer_past=layer_past,
            use_cache=use_cache,
            output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        outputs = attention_outputs[1:]

        # Residual connection.
        alpha = (2 * self.num_layers) ** 0.5
        hidden_states = attention_input * alpha + attention_output

        mlp_input = self.post_attention_layernorm(hidden_states)

        # MLP.
        mlp_output = self.mlp(mlp_input)

        # Second residual connection.
        output = mlp_input * alpha + mlp_output

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class ChatGLMPreTrainedModel(MSPreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = ChatGLMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Cell):
        """Initialize the weights."""
        return

    def get_masks(self, input_ids, device, padding_mask=None):
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
        attention_mask = ops.ones((batch_size, seq_length, seq_length))
        attention_mask.tril_()
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        if padding_mask is not None:
            attention_mask = attention_mask * padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)
        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()

        return attention_mask

    def get_position_ids(self, input_ids, device):
        MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
        seqs = input_ids.tolist()
        mask_positions, use_gmasks = [], []
        for seq in seqs:
            mask_token = gMASK if gMASK in seq else MASK
            use_gmask = mask_token == gMASK
            mask_positions.append(seq.index(mask_token))
            use_gmasks.append(use_gmask)
        batch_size, seq_length = input_ids.shape
        if use_gmasks is None:
            use_gmasks = [False] * batch_size
        context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
        if self.position_encoding_2d:
            position_ids = ops.arange(seq_length, dtype=ms.int32).unsqueeze(0).repeat(batch_size, 1)
            for i, context_length in enumerate(context_lengths):
                position_ids[i, context_length:] = mask_positions[i]
            block_position_ids = [ops.cat((
                ops.zeros(context_length, dtype=ms.int32),
                ops.arange(seq_length - context_length, dtype=ms.int32) + 1
            )) for context_length in context_lengths]
            block_position_ids = ops.stack(block_position_ids, axis=0)
            position_ids = ops.stack((position_ids, block_position_ids), axis=1)
        else:
            position_ids = ops.arange(seq_length, dtype=ms.int32).unsqueeze(0).repeat(batch_size, 1)
            for i, context_length in enumerate(context_lengths):
                if not use_gmasks[i]:
                    position_ids[i, context_length:] = mask_positions[i]

        return position_ids

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ChatGLMModel):
            module.gradient_checkpointing = value


CHATGLM_6B_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config ([`~ChatGLM6BConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CHATGLM_6B_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`ChatGLM6BTokenizer`].
            See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range `[0, config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert *input_ids* indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class ChatGLMModel(ChatGLMPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    `is_decoder` argument of the configuration set to `True`.
    To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder`
    argument and `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config: ChatGLMConfig, empty_init=True):
        super().__init__(config)
        init_method = default_init
        # recording parameters
        self.max_sequence_length = config.max_sequence_length
        self.hidden_size = config.hidden_size
        self.params_dtype = ms.half
        self.num_attention_heads = config.num_attention_heads
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.layernorm_epsilon = config.layernorm_epsilon
        self.inner_hidden_size = config.inner_hidden_size
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads
        self.position_encoding_2d = config.position_encoding_2d
        self.pre_seq_len = config.pre_seq_len
        self.prefix_projection = config.prefix_projection

        self.word_embeddings = init_method(
            nn.Embedding,
            num_embeddings=self.vocab_size, embedding_dim=self.hidden_size,
            dtype=self.params_dtype
        )
        self.gradient_checkpointing = False

        def get_layer(layer_id):
            return GLMBlock(
                self.hidden_size,
                self.num_attention_heads,
                self.layernorm_epsilon,
                layer_id,
                inner_hidden_size=self.inner_hidden_size,
                hidden_size_per_attention_head=self.hidden_size_per_attention_head,
                layernorm=nn.LayerNorm,
                use_bias=True,
                params_dtype=self.params_dtype,
                position_encoding_2d=self.position_encoding_2d,
                empty_init=empty_init
            )

        self.layers = nn.CellList(
            [get_layer(layer_id) for layer_id in range(self.num_layers)]
        )

        # Final layer norm before output.
        self.final_layernorm = nn.LayerNorm(self.hidden_size, eps=self.layernorm_epsilon)

        if self.pre_seq_len is not None:
            for param in self.parameters():
                param.requires_grad = False
            self.prefix_tokens = ops.arange(self.pre_seq_len, dtype=ms.int32)
            self.prefix_encoder = PrefixEncoder(config)
            self.dropout = nn.Dropout(p=0.1)

            # total_params = sum(p.numel() for p in self.parameters())
            # trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            # print("Using p-tuning v2: # trainable_params = {} / {}".format(trainable_params, total_params))

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: ms.Tensor):
        self.word_embeddings = new_embeddings

    def get_prompt(self, batch_size, device, dtype=ms.half):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_layers * 2,
            self.num_attention_heads,
            self.hidden_size // self.num_attention_heads
        )
        # seq_len, b, nh, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
        # past_key_values = [(v[0], v[1]) for v in past_key_values]
        return past_key_values

    def forward(
            self,
            input_ids: Optional[ms.Tensor] = None,
            position_ids: Optional[ms.Tensor] = None,
            attention_mask: Optional[ms.Tensor] = None,
            full_attention_mask: Optional[ms.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[ms.Tensor, ms.Tensor], ...]] = None,
            inputs_embeds: Optional[ms.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ms.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            logger.warning_once("Specify both input_ids and inputs_embeds at the same time, will use inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if past_key_values is None:
            if self.pre_seq_len is not None:
                past_key_values = self.get_prompt(batch_size=input_ids.shape[0], device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)
            else:
                past_key_values = tuple([None] * len(self.layers))

            if full_attention_mask is None:
                full_attention_mask = self.get_masks(
                    input_ids,
                    device=input_ids.device,
                    padding_mask=attention_mask
                )

            if position_ids is None:
                position_ids = self.get_position_ids(
                    input_ids,
                    device=input_ids.device,
                )
        else:
            if attention_mask is not None:
                full_attention_mask = (attention_mask < 0.5).bool()
                full_attention_mask = full_attention_mask.unsqueeze(1).unsqueeze(1)

        if self.pre_seq_len is not None and full_attention_mask is not None:
            prefix_attention_mask = ops.ones((batch_size, 1, input_ids.size(-1), self.pre_seq_len))
            prefix_attention_mask = (prefix_attention_mask < 0.5).bool()
            full_attention_mask = ops.cat((prefix_attention_mask, full_attention_mask), axis=3)

        # [seq_len, batch, hidden_size]
        hidden_states = inputs_embeds.transpose(0, 1)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if full_attention_mask is None:
            full_attention_mask = ops.zeros((1, 1)).bool()
        else:
            full_attention_mask = full_attention_mask.to(hidden_states.device)

        for i, layer in enumerate(self.layers):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_past = past_key_values[i]

            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    position_ids,
                    full_attention_mask,
                    ms.tensor(i),
                    layer_past,
                    use_cache,
                    output_attentions
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=full_attention_mask,
                    layer_id=ms.tensor(i),
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions
                )

            hidden_states = layer_ret[0]

            if use_cache:
                presents = presents + (layer_ret[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_ret[2 if use_cache else 1],)

        # Final layer norm.
        hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel):
    def __init__(self, config: ChatGLMConfig, empty_init=True):
        super().__init__(config)
        init_method = default_init

        # self.hidden_size = config.hidden_size
        # self.params_dtype = torch.half
        # self.vocab_size = config.vocab_size
        self.max_sequence_length = config.max_sequence_length

        self.position_encoding_2d = config.position_encoding_2d

        self.transformer = ChatGLMModel(config, empty_init=empty_init)

        self.lm_head = init_method(
            nn.Dense,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=ms.half
        )

        self.config = config

        self.quantized = False

        if self.config.quantization_bit:
            self.quantize(self.config.quantization_bit, empty_init=True)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = ops.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], axis=-1
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id[:, 1, :] += 1
            model_kwargs["position_ids"] = ops.cat(
                [position_ids, new_position_id], axis=-1
            )

        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids: ms.Tensor,
            past: Optional[ms.Tensor] = None,
            past_key_values: Optional[ms.Tensor] = None,
            attention_mask: Optional[ms.Tensor] = None,
            position_ids: Optional[ms.Tensor] = None,
            **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        if past is not None or past_key_values is not None:
            last_token = input_ids[:, -1].unsqueeze(-1)
            if position_ids is None:
                position_ids = self.get_position_ids(input_ids, device=input_ids.device)
            position_ids = position_ids[..., -1:]

            if past is None:
                past = past_key_values
            return {
                "input_ids": last_token,
                "past_key_values": past,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                **kwargs
            }
        else:
            return {
                "input_ids": input_ids,
                "past_key_values": past,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                **kwargs
            }

    def forward(
            self,
            input_ids: Optional[ms.Tensor] = None,
            position_ids: Optional[ms.Tensor] = None,
            attention_mask: Optional[ms.Tensor] = None,
            past_key_values: Optional[Tuple[ms.Tensor]] = None,
            inputs_embeds: Optional[ms.Tensor] = None,
            labels: Optional[ms.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states).permute(1, 0, 2).contiguous()

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(ms.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[ms.Tensor, ms.Tensor], ...], beam_idx: ms.Tensor
    ) -> Tuple[Tuple[ms.Tensor, ms.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(1, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )

    def process_response(self, response):
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response


    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs


    # @torch.no_grad()
    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1,
             do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        inputs = self.build_inputs(tokenizer, query, history=history)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history

    # @torch.no_grad()
    def stream_chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048,
                    do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        inputs = self.build_inputs(tokenizer, query, history=history)
        for outputs in self.stream_generate(**inputs, **gen_kwargs):
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
            response = tokenizer.decode(outputs)
            response = self.process_response(response)
            new_history = history + [(query, response)]
            yield response, new_history

    # @torch.no_grad()
    # def stream_generate(
    #         self,
    #         input_ids,
    #         generation_config: Optional[GenerationConfig] = None,
    #         logits_processor: Optional[LogitsProcessorList] = None,
    #         stopping_criteria: Optional[StoppingCriteriaList] = None,
    #         prefix_allowed_tokens_fn: Optional[Callable[[int, ms.Tensor], List[int]]] = None,
    #         **kwargs,
    # ):
    #     batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    #
    #     if generation_config is None:
    #         generation_config = self.generation_config
    #     generation_config = copy.deepcopy(generation_config)
    #     model_kwargs = generation_config.update(**kwargs)
    #     bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id
    #
    #     if isinstance(eos_token_id, int):
    #         eos_token_id = [eos_token_id]
    #
    #     has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    #     if has_default_max_length and generation_config.max_new_tokens is None:
    #         warnings.warn(
    #             f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
    #             "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
    #             " recommend using `max_new_tokens` to control the maximum length of the generation.",
    #             UserWarning,
    #         )
    #     elif generation_config.max_new_tokens is not None:
    #         generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
    #         if not has_default_max_length:
    #             logger.warn(
    #                 f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
    #                 f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
    #                 "Please refer to the documentation for more information. "
    #                 "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
    #                 UserWarning,
    #             )
    #
    #     if input_ids_seq_length >= generation_config.max_length:
    #         input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
    #         logger.warning(
    #             f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
    #             f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
    #             " increasing `max_new_tokens`."
    #         )
    #
    #     # 2. Set generation parameters if not already defined
    #     logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    #     stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    #
    #     logits_processor = self._get_logits_processor(
    #         generation_config=generation_config,
    #         input_ids_seq_length=input_ids_seq_length,
    #         encoder_input_ids=input_ids,
    #         prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    #         logits_processor=logits_processor,
    #     )
    #
    #     stopping_criteria = self._get_stopping_criteria(
    #         generation_config=generation_config, stopping_criteria=stopping_criteria
    #     )
    #     logits_warper = self._get_logits_warper(generation_config)
    #
    #     unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    #     scores = None
    #     while True:
    #         model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
    #         # forward pass to get next token
    #         outputs = self(
    #             **model_inputs,
    #             return_dict=True,
    #             output_attentions=False,
    #             output_hidden_states=False,
    #         )
    #
    #         next_token_logits = outputs.logits[:, -1, :]
    #
    #         # pre-process distribution
    #         next_token_scores = logits_processor(input_ids, next_token_logits)
    #         next_token_scores = logits_warper(input_ids, next_token_scores)
    #
    #         # sample
    #         probs = nn.functional.softmax(next_token_scores, dim=-1)
    #         if generation_config.do_sample:
    #             next_tokens = ops.multinomial(probs, num_samples=1).squeeze(1)
    #         else:
    #             next_tokens = ops.argmax(probs, dim=-1)
    #
    #         # update generated ids, model inputs, and length for next step
    #         input_ids = ops.cat([input_ids, next_tokens[:, None]], axis=-1)
    #         model_kwargs = self._update_model_kwargs_for_generation(
    #             outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
    #         )
    #         unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).to(ms.int32))
    #
    #         # stop when each sentence is finished, or if we exceed the maximum length
    #         if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
    #             break
    #         yield input_ids

    # def quantize(self, bits: int, empty_init=False, **kwargs):
    #     if bits == 0:
    #         return
    #
    #     from .quantization import quantize
    #
    #     if self.quantized:
    #         logger.info("Already quantized.")
    #         return self
    #
    #     self.quantized = True
    #
    #     self.config.quantization_bit = bits
    #
    #     self.transformer = quantize(self.transformer, bits, empty_init=empty_init, **kwargs)
    #     return self


class ChatGLMForConditionalGenerationWithImage(ChatGLMForConditionalGeneration):
    def __init__(self, config: ChatGLMConfig, empty_init=True):
        super().__init__(config, empty_init=empty_init)
        from .visual import BLIP2
        self.image_encoder = BLIP2(config.eva_config, config.qformer_config)
        self.image_length = config.image_length

    @staticmethod
    def process_image(text, image=None):
        '''Process image in text.
        Args:
            text: str, text.
            image: Optional, image path / url / PIL image.
        '''
        from .visual import BlipImageEvalProcessor
        from PIL import Image
        from io import BytesIO

        image_position = text.rfind("<img>") + 5
        # extract path from <img></img> using re
        image_path = re.findall(r"<img>(.*?)</img>", text)
        image_path = image_path[-1] if image_path else None
        if image_path is not None:
            assert image is None, "image and image_path cannot be both not None."
            text = text.replace(f"<img>{image_path}</img>", "<img></img>")
            # url
            if image_path.startswith("http"):
                response = requests.get(image_path, timeout=10)
                image = Image.open(BytesIO(response.content))
            # local path
            else:
                image = Image.open(image_path)
        if image is not None:
            processor = BlipImageEvalProcessor(224)
            image = processor(image.convert('RGB'))
            image = image.unsqueeze(0)
        return text, image_position, image

    def build_inputs_with_image(self, tokenizer, image_path: str, query: str, history: List[Tuple[str, str]] = None):
        image_path = image_path.strip()
        if image_path:
            prompt = "<img>{}</img>".format(image_path)
        else:
            prompt = ""
        for i, (old_query, response) in enumerate(history):  # history removes image urls/paths, while query does not.
            prompt += "问：{}\n答：{}\n".format(old_query, response)
        prompt += "问：{}\n答：".format(query)
        prompt, image_position, torch_image = self.process_image(prompt)
        if torch_image is not None:
            torch_image = torch_image.to(self.dtype).to(self.device)
            input0 = tokenizer.encode(prompt[:image_position], add_special_tokens=False)
            input1 = [tokenizer.unk_token_id] * self.image_length
            input2 = tokenizer.encode(prompt[image_position:], add_special_tokens=False)
            inputs = sum([input0, input1, input2], [])
            inputs = {
                "input_ids": ms.Tensor([tokenizer.build_inputs_with_special_tokens(inputs)], dtype=ms.int32).to(
                    self.device),
                "pre_image_length": len(input0),
                "images": torch_image}
        else:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = inputs.to(self.device)
            inputs["pre_image_length"] = 0
        return inputs

    @torch.no_grad()
    def chat(self, tokenizer, image_path: str, query: str, history: List[Tuple[str, str]] = None, max_length: int = 1024,
             min_length=100, do_sample=True, top_p=0.4, top_k=100, temperature=0.8, repetition_penalty=1.2, logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "min_length": min_length, "do_sample": do_sample, "top_p": top_p,
                      "top_k": top_k, "temperature": temperature, "repetition_penalty": repetition_penalty,
                      "logits_processor": logits_processor, **kwargs}
        inputs = self.build_inputs_with_image(tokenizer, image_path, query, history=history)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history


    @torch.no_grad()
    def stream_chat(self, tokenizer, image_path: str, query: str, history: List[Tuple[str, str]] = None,
                    max_length: int = 1024, min_length=100, do_sample=True, top_p=0.4, top_k=100, temperature=0.8,
                    repetition_penalty=1.2, logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "min_length": min_length, "do_sample": do_sample, "top_p": top_p,
                      "top_k": top_k, "temperature": temperature, "repetition_penalty": repetition_penalty,
                      "logits_processor": logits_processor, **kwargs}
        inputs = self.build_inputs_with_image(tokenizer, image_path, query, history=history)
        for outputs in self.stream_generate(**inputs, **gen_kwargs):
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
            response = tokenizer.decode(outputs)
            response = self.process_response(response)
            new_history = history + [(query, response)]
            yield response, new_history

    def forward(
            self,
            input_ids: Optional[ms.Tensor] = None,
            position_ids: Optional[ms.Tensor] = None,
            attention_mask: Optional[ms.Tensor] = None,
            images: Optional[ms.Tensor] = None,
            pre_image_length: Optional[int] = None,
            past_key_values: Optional[Tuple[ms.Tensor]] = None,
            inputs_embeds: Optional[ms.Tensor] = None,
            labels: Optional[ms.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        if inputs_embeds is None and past_key_values is None and images is not None:
            image_embeds = self.image_encoder(images)
            pre_id, pads, post_id = ops.tensor_split(input_ids,
                                                       [pre_image_length, pre_image_length + self.image_length],
                                                       axis=1)  # image after [Round 0]\n问：<img>
            pre_txt_emb = self.transformer.word_embeddings(pre_id)
            post_txt_emb = self.transformer.word_embeddings(post_id)
            inputs_embeds = ops.cat([pre_txt_emb, image_embeds, post_txt_emb], axis=1)
        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )