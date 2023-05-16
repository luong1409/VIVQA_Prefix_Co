import json
import os
import pickle
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F


class VDataset(Dataset):
    def __init__(
        self, 
        data_root: str,
        # json_with_parent: str,
        transforms,
        processor,
        config,
        split: str = 'train',
        seed=0,
        vocab_path='dataset/AnsVocab_from_train.txt'
    ):

        super().__init__()
        self.data_root = data_root
        self.transforms = transforms(config.image_size)
        self.processor = processor
        self.padding_mode = config.padding_mode
        
        self.split = split
        self.is_train = (split == 'train')
        self.seed = seed
        self.vocab_ans = {}
        with open(vocab_path) as f:
            lines = f.read().split('\n')
            for idx, line in enumerate(lines):
                self.vocab_ans[line] = idx
        self.vocab_ans['unk'] = len(self.vocab_ans)
        
        fn2captions_pickle, split2fns_pickle = self._assign_data()

        with open(fn2captions_pickle, 'rb') as f:
            self.fn2captions = pickle.load(f)

        with open(split2fns_pickle, 'rb') as f:
            self.split2fns = pickle.load(f)
        
        index_mapper = list()
        img_files = self.split2fns[split]
        for img_file in tqdm(img_files):
                question, answer = self.fn2captions[img_file]
                index_mapper.append({
                    'image': img_file,
                    'question': question,
                    'answer' : answer
                })

        self.index_mapper = index_mapper

        # passing some attributes to config
        # get actual token_id
        config.pad_token_id = getattr(self.processor.tokenizer, 'pad_token_id')
        config.bos_token_id = getattr(self.processor.tokenizer, 'bos_token_id')
        config.eos_token_id = getattr(self.processor.tokenizer, 'eos_token_id')
        # get actual vocab_size
        config.vocab_size = getattr(self.processor.tokenizer, 'vocab_size')


    def __len__(self) -> int:
        return len(self.index_mapper)
    

    def __getitem__(self, index):

        ques_ans = self.index_mapper[index]
        question = ques_ans['question']
        answer = ques_ans['answer']

        image = self._get_transformed_image(str(ques_ans['image']))
        # tokenize
        assert self.processor is not None
        question_ids = self._get_tokenized_text(question)
        
        if isinstance(answer, list):
            answer_ids = self._get_tokenized_answer(answer)
        else:
            answer_ids = self._get_tokenized_answer([answer])
        # answer_ids = F.one_hot(input=answer_ids, num_classes=len(self.vocab_ans))
        
        return image, question_ids, answer_ids

    def _get_tokenized_answer(self, answers):
        answer_ids = []
        for answer in answers:
            answer_ids.append(self.vocab_ans.get(answer, self.vocab_ans['unk']))
    
        return torch.tensor(answer_ids).unsqueeze(0)
    
    def _get_absolute_image_path(self, file):
        return NotImplementedError("Absolute image paths depend on dataset")

    def _get_transformed_image(self, file):
        if not file.startswith('/'):
            file = self._get_absolute_image_path(file)
            
        try:
            image = Image.open(file).convert('RGB')
        except:
            from loguru import logger
            logger.warning(f"Can not open file {file}")
            return None
        if self.transforms:
            # image = [tr(image) for tr in (self.transforms)]
            image = self.transforms(image)
        return image


    def _get_tokenized_text(self, text):

        tokenizer = self.processor.tokenizer

        text_encoding = tokenizer(
            text=text,
            add_special_tokens=False,
            padding=self.padding_mode,
            truncation=True,
            return_tensors='pt'
        )
        return text_encoding['input_ids'].squeeze(0)
    
    # @staticmethod
    def _assign_data(self, *args, **kwargs):
        raise NotImplementedError("Assigning split and image-captions pairs depends on dataset")



class VQADataset(VDataset):
    def __init__(self, split, **kwargs) -> None:
        # split = 'test' if split == 'val' else split
        if not 'data_root' in kwargs:
            kwargs['data_root'] = "../../shared/"

        super().__init__(split=split, **kwargs)
        self.prefix_text_len = 1

    def _get_absolute_image_path(self, file):
        if self.is_train:
            return os.path.join(self.data_root, f'train/{file}.jpg')
        return os.path.join(self.data_root, f'test/{file}.jpg')


    def _assign_data(
        self,
        cache_dir='./.cache/custom_assigned/',
        # TODO: Change to locate csv file
        train_csv= "./train.csv",
        test_csv= "./test.csv",
    ):
        fn2captions_pickle = os.path.join(cache_dir, 'fn2captions.pickle')
        split2fns_pickle = os.path.join(cache_dir, 'split2fns.pickle')
        
        if os.path.exists(fn2captions_pickle) and os.path.exists(split2fns_pickle):
            print('Using cache')
            return fn2captions_pickle, split2fns_pickle
        
        train = pd.read_csv(train_csv, sep=',')
        test = pd.read_csv(test_csv, sep=',')
        
        # list of dicts {'filepath'}
        fn2captions = defaultdict(list)
        split2fns = {
            'train': [],
            'test': []
        }

        split2fns['train'] = train.values[:, 3].tolist()
        # split2fns['test'] = test[:, 1].tolist()
        split2fns['test'] = test.values[:, 3].tolist()
        
        for row in tqdm(train.iterrows()):
            fn2captions[row[1].img_id] = [row[1].question, row[1].answer]
        for row in tqdm(test.iterrows()):
            fn2captions[row[1].img_id] = [row[1].question, row[1].answer]

        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        with open(fn2captions_pickle, 'wb') as f:
            pickle.dump(fn2captions, f)

        with open(split2fns_pickle, 'wb') as f:
            pickle.dump(split2fns, f)

        return fn2captions_pickle, split2fns_pickle
        