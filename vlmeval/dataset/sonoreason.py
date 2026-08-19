import os

import pandas as pd

from .image_base import ImageBaseDataset
from ..smp import load, dump


class SonoReasonDD(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {'sonoreason_dd_breast': ''}   # local TSV in LMUData
    DATASET_MD5 = {'sonoreason_dd_breast': ''}
    PROMPT_COLUMNS = {
        'direct_diagnosis': 'direct_prompt',
        'reasoning_diagnosis': 'reasoning_prompt',
    }

    def load_data(self, dataset):
        data_root = os.environ.get('LMUData', os.path.expanduser('~/LMUData'))
        relative_path = os.environ.get('SONOREASON_DATASET_FILE', f'{dataset}.tsv')
        path = os.path.join(data_root, relative_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'SonoReason dataset file not found: {path}. '
                'Set LMUData to the dataset root and SONOREASON_DATASET_FILE '
                'to the relative TSV path.'
            )

        prompt_strategy = os.environ.get(
            'SONOREASON_PROMPT_STRATEGY', 'direct_diagnosis')
        if prompt_strategy not in self.PROMPT_COLUMNS:
            choices = ', '.join(sorted(self.PROMPT_COLUMNS))
            raise ValueError(
                f'Unsupported SonoReason prompt strategy {prompt_strategy!r}. '
                f'Choose one of: {choices}.'
            )

        data = pd.read_csv(path, sep='\t')
        prompt_column = self.PROMPT_COLUMNS[prompt_strategy]
        required_columns = {'img_data', prompt_column, 'class_label'}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(
                f'SonoReason TSV is missing required columns: {sorted(missing_columns)}')

        # Map the published SonoReason schema to VLMEvalKit's canonical fields.
        data = data.rename(columns={
            'img_data': 'image',
            prompt_column: 'question',
            'class_label': 'answer',
        })
        data['index'] = range(len(data))
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt = self.dump_image(line)              # decodes base64 -> image path(s)
        msgs = []
        if isinstance(tgt, list):
            msgs += [dict(type='image', value=p) for p in tgt]
        else:
            msgs.append(dict(type='image', value=tgt))
        msgs.append(dict(type='text', value=line['question']))
        return msgs

    def evaluate(self, eval_file, **kwargs):
        data = load(eval_file)
        CLASSES = ['malignant', 'benign', 'normal']
        def norm(x):
            s = str(x).lower()
            for c in CLASSES:
                if c in s: return c
            return s.strip()
        data['hit'] = [norm(p) == norm(a) for p, a in zip(data['prediction'], data['answer'])]
        acc = data['hit'].mean()
        res = pd.DataFrame([{'accuracy': acc, 'n': len(data)}])
        dump(res, eval_file.replace('.xlsx', '_acc.csv'))
        return res
