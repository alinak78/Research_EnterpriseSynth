from pathlib import Path
import hashlib,json,platform,sys,warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from enterprise_audit.study import *
ROOT=Path(__file__).resolve().parents[1]
def main():
 if (ROOT/'results/metrics.csv').exists():raise SystemExit('Refusing to overwrite results')
 files=sorted(ROOT.glob('src/**/*.py'))+[ROOT/'paper/STUDY_PROTOCOL.md',Path(__file__)]+sorted(ROOT.glob('data/raw/*.zip'))
 (ROOT/'results/frozen_inputs.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
 rows=[];bias=[];splits={};audit=[]
 for di,name in enumerate(['adult','bank','wine']):
  df,cats,original=load_data(ROOT,name)
  audit.append(dict(dataset=name,source_rows=original,analysis_rows=len(df),positives=int(df.target.sum()),real_violations=int((~valid_rows(df,name)).sum())))
  for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=2027).split(df,df.target)):
   splits[f'{name}_{fold}']={'train':tr.tolist(),'test':te.tolist()}
   train,test=df.iloc[tr].copy(),df.iloc[te].copy();pre=encode(train,cats);n=len(train);seed=2027+di*100+fold
   def evaluate(release,generator,arm,examined):
    tag=f'{name}_{fold}_{generator}_{arm}'
    if len(release)!=n:raise RuntimeError(f'Incomplete release {tag}: {len(release)}/{n}')
    met,score,pred=predict(release,test,pre)
    rows.append(dict(dataset=name,fold=fold,generator=generator,arm=arm,seed=seed,n_train=n,n_test=len(test),n_release=len(release),examined=examined,candidate_ratio=examined/n,**met,**quality(release,train,name,cats)))
    pd.DataFrame({'row_id':te,'truth':test.target.to_numpy(),'score':score,'prediction':pred}).to_csv(ROOT/f'results/{tag}_predictions.csv',index=False)
    if generator!='real':release.to_csv(ROOT/f'data/generated/{tag}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
   evaluate(train,'real','reference',n)
   evaluate(train.iloc[np.random.default_rng(seed).integers(0,n,n)].copy(),'bootstrap','reference',n)
   for gi,kind in enumerate(['marginal','joint_copula','conditional_copula']):
    with warnings.catch_warnings():
     warnings.filterwarnings('ignore',message='invalid value encountered in divide',category=RuntimeWarning)
     pool=Generator(train,cats,kind).sample(10*n,seed+10000*(gi+1))
    valid=valid_rows(pool,name);p=float(pool.target.mean());a0=float(valid[pool.target.to_numpy()==0].mean());a1=float(valid[pool.target.to_numpy()==1].mean())
    bias.append(dict(dataset=name,fold=fold,generator=kind,pool_positive=p,acceptance_negative=a0,acceptance_positive=a1,valid_pool_positive=float(pool.loc[valid,'target'].mean()),identity_prediction=p*a1/(p*a1+(1-p)*a0),pool_sha256=hashlib.sha256(pool.to_csv(index=False).encode()).hexdigest()))
    for arm in ['raw','project','reject','quota']:
     release,examined=process_pool(pool,name,arm,n,float(train.target.mean()));evaluate(release,kind,arm,examined)
   print(name,fold,'complete',flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'results/metrics.csv',index=False)
 pd.DataFrame(bias).to_csv(ROOT/'results/selection_bias.csv',index=False)
 pd.DataFrame(audit).to_csv(ROOT/'results/data_audit.csv',index=False)
 (ROOT/'results/splits.json').write_text(json.dumps(splits))
 (ROOT/'results/environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__),indent=2))
 pd.DataFrame(rows).groupby(['dataset','generator','arm']).agg({x:['mean','std','min','max'] for x in ['f1','recall','average_precision','violation_rate','prevalence_drift','marginal_distance','candidate_ratio','exact_train_match']}).to_csv(ROOT/'results/summary.csv')
if __name__=='__main__':main()
