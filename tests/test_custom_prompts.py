from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from wastevlm.custom_prompts import read_prompt_csv


def test_balanced_custom_prompt_import_and_missing_category():
    rows=[dict(family=f,variant=v,category=c,prompt_index=0,prompt=f'a photo of {c}')
          for f in ['name_preserving','object_expressions'] for v in [1,2] for c in ['a','b']]
    with TemporaryDirectory() as d:
        p=Path(d)/'prompts.csv'
        pd.DataFrame(rows).to_csv(p,index=False)
        assert len(read_prompt_csv(p,['a','b']))==4
        pd.DataFrame(rows[:-1]).to_csv(p,index=False)
        try:read_prompt_csv(p,['a','b'])
        except ValueError:pass
        else:raise AssertionError('Unbalanced class treatment accepted')
