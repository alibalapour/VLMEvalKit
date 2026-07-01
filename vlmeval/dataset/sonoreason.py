import os, re, pandas as pd
from .image_base import ImageBaseDataset
from ..smp import load, dump

class SonoReasonDD(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {'sonoreason_dd_breast': ''}   # local TSV in LMUData
    DATASET_MD5 = {'sonoreason_dd_breast': ''}

    def load_data(self, dataset):
        import os, pandas as pd
        data_root = os.environ.get('LMUData', os.path.expanduser('~/LMUData'))
        path = os.path.join(data_root, f'{dataset}.tsv')
        return pd.read_csv(path, sep='\t')

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
