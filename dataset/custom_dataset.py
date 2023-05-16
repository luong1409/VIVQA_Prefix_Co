import json
import os
import pickle
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from .prefix_dataset import PrefixDataset
import pandas as pd
from sklearn.model_selection import train_test_split

class CustomDataset(PrefixDataset):
    def __init__(self, split, **kwargs) -> None:
        if not 'data_root' in kwargs:
            kwargs['data_root'] = "../../shared/"
        super().__init__(split=split, **kwargs)
        self.prefix_text_len = 1

    def _random_prefix_text_len(self):
        assert self.random_prefix_len
        
        random.seed(self.seed)
        return random.choice(range(2, 8))

    def _get_absolute_image_path(self, file):
        return os.path.join(self.data_root, file)


    def _assign_data(
        self,
        cache_dir='./.cache/custom_assigned/IC/',
        # TODO: Change to locate csv file
        csv_with_parent= "./segmented_IC.csv"
    ):

        fn2captions_pickle = os.path.join(cache_dir, 'fn2captions.pickle')
        split2fns_pickle = os.path.join(cache_dir, 'split2fns.pickle')
        
        if os.path.exists(fn2captions_pickle) and os.path.exists(split2fns_pickle):
            print('Using cache')
            return fn2captions_pickle, split2fns_pickle
        
        df = pd.read_csv(csv_with_parent, sep='\t')
        
        # TODO: change split strategy
        train, val = train_test_split(df.values, test_size=0.2, random_state=self.seed)
        # val, test = train_test_split(test, test_size=0.5, random_state=self.seed)
        
        # list of dicts {'filepath'}
        fn2captions = defaultdict(list)
        split2fns = {
            'train': [],
            'val': []
        }

        split2fns['train'] = train[:, 1].tolist()
        split2fns['val'] = val[:, 1].tolist()
        
        for row in tqdm(df.iterrows()):
            fn2captions[row[1].image] = [row[1].caption]

        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        with open(fn2captions_pickle, 'wb') as f:
            pickle.dump(fn2captions, f)

        with open(split2fns_pickle, 'wb') as f:
            pickle.dump(split2fns, f)

        return fn2captions_pickle, split2fns_pickle
        