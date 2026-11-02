"""Recompute published metrics from retained evidence without fitting models."""
from pathlib import Path
import hashlib,json,re
import numpy as np,pandas as pd
from sklearn.metrics import f1_score,recall_score,balanced_accuracy_score,average_precision_score,accuracy_score
from enterprise_audit.study import load_data,valid_rows
R=Path(__file__).resolve().parents[1]
def validate():
 frozen=json.loads((R/'results/frozen_inputs.json').read_text())
 for name,digest in frozen.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
 m=pd.read_csv(R/'results/metrics.csv');assert len(m)==210
 assert len(list((R/'data/generated').glob('*.csv.gz')))==195
 assert not m.duplicated(['dataset','fold','generator','arm']).any()
 splits=json.loads((R/'results/splits.json').read_text())
 for dataset in ['adult','bank','wine']:
  df,_,_=load_data(R,dataset);held=[]
  for fold in range(5):
   ids=splits[f'{dataset}_{fold}'];assert not set(ids['train'])&set(ids['test']);held+=ids['test']
  assert sorted(held)==list(range(len(df)))
 for row in m.itertuples():
  tag=f'{row.dataset}_{row.fold}_{row.generator}_{row.arm}'
  p=pd.read_csv(R/f'results/{tag}_predictions.csv');ids=splits[f'{row.dataset}_{row.fold}']
  df,_,_=load_data(R,row.dataset)
  assert p.row_id.tolist()==ids['test'];assert np.array_equal(p.truth,df.iloc[ids['test']].target)
  for key,fun in [('f1',f1_score),('recall',recall_score),('balanced_accuracy',balanced_accuracy_score),('accuracy',accuracy_score)]:
   assert np.isclose(fun(p.truth,p.prediction),getattr(row,key)),(tag,key)
  assert np.isclose(average_precision_score(p.truth,p.score),row.average_precision)
  if row.generator!='real':
   s=pd.read_csv(R/f'data/generated/{tag}.csv.gz');assert len(s)==row.n_train
   assert np.isclose(1-valid_rows(s,row.dataset).mean(),row.violation_rate)
   assert np.isclose(s.target.mean(),row.positive_fraction)
   if row.arm=='project':
    raw=pd.read_csv(R/f'data/generated/{row.dataset}_{row.fold}_{row.generator}_raw.csv.gz')
    assert np.array_equal(s.target,raw.target)
   if row.arm=='quota':assert s.target.sum()==int(df.iloc[ids['train']].target.sum())
 b=pd.read_csv(R/'results/selection_bias.csv');assert np.allclose(b.valid_pool_positive,b.identity_prediction)
 tex=(R/'paper/main.tex').read_text();bib=(R/'paper/references.bib').read_text()
 cites=set(re.findall(r'\\cite\{([^}]+)\}',tex));keys=set(re.findall(r'@\w+\{([^,]+),',bib));assert cites==keys,(cites,keys)
 for fragment in ['utility_table.tex','audit_table.tex']:
  for line in (R/'paper'/fragment).read_text().splitlines():assert line in tex,(fragment,line)
 print('PASS: frozen inputs; 210 predictions/metrics; 195 releases; held-out split coverage; rules; quotas; citations')
if __name__=='__main__':validate()
