from __future__ import annotations
import io,zipfile,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm,rankdata,wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score,recall_score,balanced_accuracy_score,average_precision_score,accuracy_score

EDUCATION={name:i+1 for i,name in enumerate(['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th','HS-grad','Some-college','Assoc-voc','Assoc-acdm','Bachelors','Masters','Prof-school','Doctorate'])}
ADULT=['age','workclass','fnlwgt','education','education_num','marital','occupation','relationship','race','sex','capital_gain','capital_loss','hours','country','target']

def load_data(root:Path,name:str):
    with zipfile.ZipFile(root/f'data/raw/{name}.zip') as z:
        if name=='adult':
            df=pd.read_csv(io.BytesIO(z.read('adult.data')),header=None,names=ADULT,skipinitialspace=True,na_values='?')
            original=len(df);df=df.dropna().head(6000).copy()
            df['target']=(df['target'].str.strip()=='>50K').astype(int)
            categorical=['workclass','education','marital','occupation','relationship','race','sex','country']
        elif name=='bank':
            with zipfile.ZipFile(io.BytesIO(z.read('bank.zip'))) as inner:
                df=pd.read_csv(io.BytesIO(inner.read('bank.csv')),sep=';')
            original=len(df);df=df.rename(columns={'y':'target'});df['target']=(df['target']=='yes').astype(int)
            categorical=list(df.select_dtypes(include='object').columns)
        else:
            df=pd.read_csv(io.BytesIO(z.read('winequality-red.csv')),sep=';');original=len(df)
            df['target']=(df.pop('quality')>=7).astype(int);categorical=[]
    df=df.reset_index(drop=True)
    for col in categorical:df[col]=df[col].astype(str).str.strip()
    return df,categorical,original

def valid_rows(df:pd.DataFrame,name:str)->np.ndarray:
    if name=='adult':return (df.education.map(EDUCATION)==df.education_num).to_numpy()
    if name=='bank':return (((df.pdays==-1)==(df.previous==0)) & (df.previous>=0)).to_numpy()
    return (df['total sulfur dioxide']>=df['free sulfur dioxide']).to_numpy()

def project(df:pd.DataFrame,name:str)->pd.DataFrame:
    out=df.copy()
    if name=='adult':out['education_num']=out.education.map(EDUCATION).astype(float)
    elif name=='bank':out['previous']=np.where(out.pdays==-1,0,np.maximum(1,out.previous))
    else:out['total sulfur dioxide']=np.maximum(out['total sulfur dioxide'],out['free sulfur dioxide'])
    assert np.array_equal(out.target,df.target)
    assert valid_rows(out,name).all()
    return out

class Generator:
    def __init__(self,train:pd.DataFrame,categorical:list[str],kind:str):
        self.columns=list(train.columns);self.kind=kind;self.train=train
        self.codecs={c:sorted(train[c].unique()) for c in categorical}
        x=train.copy()
        for c,levels in self.codecs.items():x[c]=x[c].map({v:i for i,v in enumerate(levels)})
        self.values=x.to_numpy(dtype=float);self.target_index=self.columns.index('target')
        self.models={}
        if kind=='conditional_copula':
            for label in [0,1]:
                rows=self.values[self.values[:,self.target_index]==label]
                self.models[label]=self.fit_copula(np.delete(rows,self.target_index,axis=1))
        elif kind=='joint_copula':self.models[None]=self.fit_copula(self.values)
    @staticmethod
    def fit_copula(x):
        z=np.column_stack([norm.ppf((rankdata(x[:,j])-.5)/len(x)) for j in range(x.shape[1])])
        corr=np.corrcoef(z,rowvar=False);corr=np.nan_to_num(corr,nan=0.);np.fill_diagonal(corr,1.)
        corr=.95*corr+.05*np.eye(x.shape[1])
        eig,vec=np.linalg.eigh(corr);factor=vec@np.diag(np.sqrt(np.maximum(eig,1e-8)))
        return np.sort(x,axis=0),factor
    @staticmethod
    def draw_copula(model,n,rng):
        sorted_values,factor=model;u=norm.cdf(rng.normal(size=(n,sorted_values.shape[1]))@factor.T)
        idx=np.minimum((u*len(sorted_values)).astype(int),len(sorted_values)-1)
        return np.column_stack([sorted_values[idx[:,j],j] for j in range(sorted_values.shape[1])])
    def sample(self,n,seed):
        rng=np.random.default_rng(seed)
        if self.kind=='marginal':x=np.column_stack([rng.choice(self.values[:,j],n) for j in range(self.values.shape[1])])
        elif self.kind=='joint_copula':x=self.draw_copula(self.models[None],n,rng)
        else:
            labels=(rng.random(n)<self.train.target.mean()).astype(int)
            x=np.zeros((n,self.values.shape[1]));x[:,self.target_index]=labels
            cols=[i for i in range(x.shape[1]) if i!=self.target_index]
            for label in [0,1]:
                mask=np.where(labels==label)[0]
                x[np.ix_(mask,cols)]=self.draw_copula(self.models[label],len(mask),rng)
        out=pd.DataFrame(x,columns=self.columns)
        for c,levels in self.codecs.items():out[c]=[levels[int(round(v))] for v in out[c]]
        out['target']=out.target.astype(int)
        return out

def process_pool(pool,name,arm,n,positive_fraction):
    mask=valid_rows(pool,name)
    if arm=='raw':return pool.iloc[:n].copy(),n
    if arm=='project':return project(pool.iloc[:n],name),n
    selected=[];quota={1:int(round(n*positive_fraction))};quota[0]=n-quota[1];counts={0:0,1:0}
    examined=0
    for i,good in enumerate(mask):
        examined=i+1
        if not good:continue
        label=int(pool.iloc[i].target)
        if arm=='quota' and counts[label]>=quota[label]:continue
        selected.append(i);counts[label]+=1
        if len(selected)==n:break
    return pool.iloc[selected].copy(),examined

def encode(train,categorical):
    num=[c for c in train.columns if c not in categorical+['target']]
    pre=ColumnTransformer([('numeric',StandardScaler(),num),('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical)])
    pre.fit(train.drop(columns='target'))
    return pre

def predict(synth,test,pre):
    start=time.perf_counter()
    y=synth.target.to_numpy()
    model=LogisticRegression(C=1.,max_iter=1000,class_weight='balanced',solver='lbfgs') if len(np.unique(y))>1 else DummyClassifier(strategy='most_frequent')
    model.fit(pre.transform(synth.drop(columns='target')),y)
    probs=model.predict_proba(pre.transform(test.drop(columns='target')))
    score=probs[:,list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(test))
    pred=(score>=.5).astype(int);truth=test.target.to_numpy()
    return {'f1':f1_score(truth,pred,zero_division=0),'recall':recall_score(truth,pred,zero_division=0),'balanced_accuracy':balanced_accuracy_score(truth,pred),'average_precision':average_precision_score(truth,score),'accuracy':accuracy_score(truth,pred),'fit_seconds':time.perf_counter()-start},score,pred

def quality(synth,train,name,categorical):
    distances=[]
    for c in train.columns:
        if c=='target':continue
        if c in categorical:
            p=train[c].value_counts(normalize=True);q=synth[c].value_counts(normalize=True)
            distances.append(sum(abs(p.get(v,0)-q.get(v,0)) for v in set(p.index)|set(q.index))/2)
        else:
            scale=np.subtract(*np.percentile(train[c],[75,25])) or np.ptp(train[c].to_numpy()) or 1.
            distances.append(wasserstein_distance(train[c],synth[c])/scale)
    originals={tuple(r) for r in train.itertuples(index=False,name=None)}
    return {'violation_rate':float(1-valid_rows(synth,name).mean()),'positive_fraction':float(synth.target.mean()),'prevalence_drift':float(abs(synth.target.mean()-train.target.mean())),'marginal_distance':float(np.mean(distances)),'exact_train_match':float(np.mean([tuple(r) in originals for r in synth.itertuples(index=False,name=None)]))}
