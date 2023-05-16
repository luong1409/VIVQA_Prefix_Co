""" MPLUG model configuration """
import os
from typing import Any, Dict, Union

import yaml
from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger()


class MPlugConfig(PretrainedConfig):

    model_type = 'mplug'

    def __init__(
            self,
            bert_config='config_bert.json',
            image_res=504,
            batch_size_train=128,
            vision_width=1024,
            distill=True,
            clip_name='ViT-L-14',  # ViT-B-16 | ViT-L-14
            batch_size_test=64,
            k_test=128,
            alpha=0.4,
            warm_up=True,
            eos='[SEP]',
            optimizer=None,
            schedular=None,
            min_length=1,
            max_length=10,
            beam_size=5,
            add_ocr=False,
            add_object=False,
            text_encoder='bert-base-uncased',
            text_decoder='bert-base-uncased',
            # clip
            clip_embed_dim=768,
            clip_image_resolution=224,
            clip_vision_layers=24,
            clip_vision_width=1024,
            clip_vision_patch_size=14,
            clip_context_length=77,
            clip_vocab_size=49408,
            clip_transformer_width=768,
            clip_transformer_heads=12,
            clip_transformer_layers=12,
            
            # SimVLM
            hidden_size=768,
            num_encoder_layers=12,
            num_decoder_layers=12,
            num_attention_heads=12,
            intermediate_size=768,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            is_encoder_decoder=True,
            image_size=224,
            patch_size=16,
            num_channels=3,
            qkv_bias=True,
            encoder_stride=16,
            image_token_len=196,
            encoder_input_len=256,
            decoder_input_len=256,
            prefix_text_len=4,
            max_prefix_text_len=60,
            vocab_size=64000,
            max_position_embeddings=60,
            padding_mode="max_length",
            max_tgt_text_len=60,
            bos_token_id=0,
            pad_token_id=1,
            eos_token_id=2,
            freeze_image=True,
            freeze_text=True,
            **kwargs):

        super().__init__(**kwargs)
        self.bert_config = bert_config
        self.image_res = image_res
        self.batch_size_train = batch_size_train
        self.vision_width = vision_width
        self.distill = distill
        self.clip_name = clip_name
        self.batch_size_test = batch_size_test
        self.k_test = k_test
        self.alpha = alpha
        self.warm_up = warm_up
        self.eos = eos
        self.optimizer = optimizer
        self.schedular = schedular
        self.min_length = min_length
        self.max_length = max_length
        self.beam_size = beam_size
        self.add_ocr = add_ocr
        self.add_object = add_object
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        # clip
        self.clip_embed_dim = clip_embed_dim
        self.clip_image_resolution = clip_image_resolution
        self.clip_vision_layers = clip_vision_layers
        self.clip_vision_width = clip_vision_width
        self.clip_vision_patch_size = clip_vision_patch_size
        self.clip_context_length = clip_context_length
        self.clip_vocab_size = clip_vocab_size
        self.clip_transformer_width = clip_transformer_width
        self.clip_transformer_heads = clip_transformer_heads
        self.clip_transformer_layers = clip_transformer_layers
        
        # SimVLM
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_encoder_decoder = is_encoder_decoder
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride
        self.image_token_len = image_token_len
        self.encoder_input_len = encoder_input_len
        self.decoder_input_len = decoder_input_len
        self.prefix_text_len = prefix_text_len
        self.max_prefix_text_len = max_prefix_text_len
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.padding_mode = padding_mode
        self.max_tgt_text_len = max_tgt_text_len
        self.freeze_image = freeze_image
        self.freeze_text = freeze_text
    

    @classmethod
    def from_yaml_file(cls, yaml_file: Union[str,
                                             os.PathLike]) -> Dict[str, Any]:
        with open(yaml_file, 'r', encoding='utf-8') as reader:
            config_dict = yaml.load(reader, Loader=yaml.Loader)
        return cls(**config_dict)