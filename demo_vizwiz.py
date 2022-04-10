from random import shuffle
import gradio as gr
import torch
import copy
import time
import requests
import io
import numpy as np
import re

import ipdb

from PIL import Image

from vilt.config import ex
from vilt.datasets.base_dataset import BaseDataset
from vilt.modules import ViLTransformerSS

from vilt.modules.objectives import cost_matrix_cosine, ipot
from vilt.transforms import pixelbert_transform
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer
from vilt.transforms import keys_to_transforms
from einops import rearrange
import torch.nn.functional as F
from transformers import DataCollatorForWholeWordMask
from functools import partial

def compute_irtr(pl_module, batch, text_encoding):
    is_training_phase = pl_module.training

    _bs, _c, _h, _w = batch["image"][0].shape
    # false_len = pl_module.hparams.config["draw_false_text"]
    # text_ids = torch.stack(
    #     [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    # )
    # text_masks = torch.stack(
    #     [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    # )
    # text_labels = torch.stack(
    #     [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    # )

    # text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    # text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    # text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    # images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    images = [x.cuda() for x in batch["image"]]
    text_ids = text_encoding["input_ids"].expand(_bs, -1)
    text_masks = text_encoding["attention_mask"].expand(_bs, -1)
    text_labels = torch.full_like(text_ids, -100)
    infer = pl_module.infer(
        {
            "image": images,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "text_labels": text_labels
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]

    return score

class ImageDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        return self.get_image(index)

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
    
    transform_keys = (
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
    )

    dataset = ImageDataset(
        data_dir=_config["data_root"],
        transform_keys=transform_keys,
        image_size=_config["image_size"],
        names=["vizwiz_test"],
        image_only=True
    )
    
    mlm_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=_config["mlm_prob"])
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False,
                                             num_workers=4, pin_memory=True, collate_fn=partial(dataset.collate, mlm_collator=mlm_collator))
    model = ViLTransformerSS(_config)
    model.setup("test")
    model.eval()

    device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
    model.to(device)

    @torch.no_grad()
    def infer(text):
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=_config["max_text_len"],
            return_special_tokens_mask=True,
            return_tensors="pt"
        ).to(device)
        scores = []
        for batch in dataloader:
            score = compute_irtr(model, batch, encoding)
            scores.append(score)
        scores = torch.cat(scores, dim=0)
        _, indices = torch.topk(scores, k=10, largest=True)
        indices = indices.cpu().tolist()
        return [dataset.get_raw_image(i) for i in indices]

    inputs = [
        gr.inputs.Textbox(label="Caption with [MASK] tokens to be filled.", lines=5),
    ]
    outputs = [
        gr.outputs.Carousel(components=["image"])
    ]

    interface = gr.Interface(
        fn=infer,
        inputs=inputs,
        outputs=outputs,
        examples=[
            ["a car"]
        ],
    )

    interface.launch(debug=True, server_name="0.0.0.0", server_port=8888)