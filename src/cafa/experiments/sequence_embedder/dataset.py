import re
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class ProteinSequenceDataset(Dataset):
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer,
        max_sequence_len: int,
        train: bool,
        model_name: str,
    ):
        super().__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.train = train
        self.model_name = model_name

    def __getitem__(self, index):
        example = self.examples[index]
        seq = example['sequence']
       

        if self.model_name.startswith('Rostlab/ProstT5_fp16'):
            seq = seq[-self.max_sequence_len :]
            # https://huggingface.co/Rostlab/ProstT5
            # replace all rare/ambiguous amino acids by X (3Di sequences does not have those)
            # and introduce white-space between all sequences (AAs and 3Di)
            seq = " ".join(list(re.sub(r'[UZOB]', 'X', seq)))

            # add pre-fixes accordingly (this already expects 3Di-sequences to be lower-case)
            # if you go from AAs to 3Di (or if you want to embed AAs), you need to prepend "<AA2fold>"
            # if you go from 3Di to AAs (or if you want to embed 3Di), you need to prepend "<fold2AA>"
            seq = (
                "<AA2fold>" + " " + seq
                if seq.isupper()
                else "<fold2AA>" + " " + seq
            )
        elif self.model_name.startswith('facebook/esm2'):
            pass
        else:
            raise ValueError(f'Unknown model_name={self.model_name}')

        tokenizer_output = self.tokenizer(
            seq,
            add_special_tokens=True,
            padding=False,
            max_length=self.max_sequence_len,
            truncation=True,
            return_tensors=None,
            return_attention_mask=False,
        )

        item = {
            'input_ids': tokenizer_output['input_ids'],
            'protein_id': str(example['protein_id']),
        }
        if 'target' in example:
            target = torch.from_numpy(example['target'].toarray()[0])
            item['target'] = target.to(dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.examples)
