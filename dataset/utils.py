from transformers import AutoTokenizer, PhobertTokenizer

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, ConcatDataset

from dataset.custom_dataset import CustomDataset
from dataset.vqa_dataset import VQADataset

from copy import deepcopy

from loguru import logger

DATASETS = {
    'custom': CustomDataset,
    'vqa': VQADataset
}

TOKENIZER_MAP = {
    'BartTokenizer': 'facebook/bart-base',
    'PhobertTokenizer': 'vinai/phobert-base'
}

def create_tokenizer(tokenizer_name):
    """Creates a seq2seq tokenizer using a pretrained model from Hugging Face.

    Args:
        tokenizer_name: The name of the tokenizer to create. Must be one of:
            - 'BartTokenizer': Bart base model
            - 'BertGenerationTokenizer': Bert for sequence generation base model
            - 'RobertaTokenizer': RoBERTa base model

    Returns:
        A seq2seq tokenizer object.
    """
    

    if not tokenizer_name in TOKENIZER_MAP:
        raise ValueError("Invalid seq2seq tokenizer name")

    return AutoTokenizer.from_pretrained(TOKENIZER_MAP[tokenizer_name])


def create_dataset(
    dataset_names=['custom'],
    split='train',
    **kwargs
):
    dataset = DATASETS[dataset_names[0]](split, **kwargs)
    return dataset

# def create_dataset(
#     dataset_names=['coco', 'vg'],
#     split='train',
#     **kwargs
# ):
#     datasets = [DATASETS[dataset_name](split, **kwargs) for dataset_name in dataset_names]
#     mixed_dataset = ConcatDataset(datasets)
#     return mixed_dataset


def collate_batch(
    batch,
    bos_token_id,
    eos_token_id,
):
    tokenizer = PhobertTokenizer.from_pretrained(
        'vinai/phobert-base'
    )
    
    images, prefix_texts, tgt_texts, label_texts, groundtruths= [], [], [], [], []
    for image, prefix_text, tgt_text, groundtruh in batch:
        
        images.append(image)
        
        prefix_texts.append(' '.join(prefix_text))
        tgt_texts.append(' '.join(tgt_text))
        
        # prefix_texts.append(prefix_text_ids)
        # drop eos in decoder input and bos in label
        # decoder_input_texts.append(
        #     tgt_text_ids[tgt_text_ids != eos_token_id]
        # )
        # label_texts.append(
        #     tgt_text_ids[tgt_text_ids != bos_token_id]
        # )
        groundtruths.append(groundtruh)

    prefix_text_ids = tokenizer(
        text=prefix_texts,
        # stick to origin model architecture, no eos or bos tokens are added to prefix text
        add_special_tokens=False,
        padding='max_length',
        truncation=True,
        max_length=60,
        return_tensors='pt'
    )

    tgt_text_ids = tokenizer(
        text=tgt_texts,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        # pad one more for later droppig bos of input and eos of label
        max_length=256 + 1,
        return_tensors='pt'
    )
    decoder_input_texts = deepcopy(tgt_text_ids)
    label_texts = deepcopy(tgt_text_ids)
    
    # logger.debug(f"eos token is {eos_token_id}")
    # logger.debug(f"bos token is {bos_token_id}")
    # logger.debug(f"Shape of target: {tgt_text_ids.input_ids.shape}")
    
    decoder_input_texts['input_ids'] = tgt_text_ids.input_ids[tgt_text_ids.input_ids != eos_token_id].reshape(tgt_text_ids.input_ids.shape[0], -1)
    label_texts['input_ids'] = tgt_text_ids.input_ids[tgt_text_ids.input_ids != bos_token_id].reshape(tgt_text_ids.input_ids.shape[0], -1)
    
    batch_items = {
        'image': torch.stack(images),
        'prefix_text': prefix_text_ids,
        'decoder_input_text': decoder_input_texts,
        'label_text': label_texts,
        'groundtruth': groundtruths
    }

    return batch_items

def collate_batch_vqa(
    batch,
    bos_token_id,
    eos_token_id,
):  
    images, questions, answers = [], [], []
    for image, question, answer in batch:
        if len(question) <= 1:
            continue
        elif image is None:
            continue
        images.append(image)
        # drop eos in decoder input and bos in label
        answers.append(answer)
        questions.append(
            question[question != bos_token_id]
        )

    image, question, answer = None, None, None
    try:
        image = torch.stack(images)
        question = torch.stack(questions)
        answer = torch.stack(answers)
    except:
        image = torch.tensor([])
        question = torch.tensor([])
        answer = torch.tensor([])
    batch_items = {
        'image': image,
        'question': question,
        'answer': answer,
    }

    return batch_items
    

def create_dataloader(
    batch_size: int,
    dataset: Dataset,
    split='train',
    sampler=None,
    num_workers=4,
    pin_memory=True
):

    def collate_fn(batch):
        if isinstance(dataset, ConcatDataset):
            bos_token_id = getattr(dataset.datasets[0].processor.tokenizer, 'bos_token_id')
            eos_token_id = getattr(dataset.datasets[0].processor.tokenizer, 'eos_token_id')
        else:
            bos_token_id = getattr(dataset.processor.tokenizer, 'bos_token_id')
            eos_token_id = getattr(dataset.processor.tokenizer, 'eos_token_id')
        
        if isinstance(dataset, VQADataset):
            return collate_batch_vqa(
                batch=batch,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id
            )
        return collate_batch(
            batch,
            bos_token_id,
            eos_token_id
        )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler if sampler else RandomSampler(dataset),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=(split == 'train')
    )
    return loader
    