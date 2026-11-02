from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];m=pd.read_csv(R/'results/metrics.csv');b=pd.read_csv(R/'results/selection_bias.csv')
assert len(m)==210 and len(b)==45
assert np.allclose(b.valid_pool_positive,b.identity_prediction)
rng=np.random.default_rng(2027);effects=[]
for (d,g),t in m[m.generator.isin(['marginal','joint_copula','conditional_copula'])].groupby(['dataset','generator']):
 p=t.pivot(index='fold',columns='arm',values='f1')
 for arm,ref in [('project','raw'),('reject','raw'),('quota','reject')]:
  delta=(p[arm]-p[ref]).to_numpy()*100;boot=delta[rng.integers(0,5,(2000,5))].mean(axis=1)
  effects.append(dict(dataset=d,generator=g,comparison=arm+'-'+ref,mean_pp=delta.mean(),min_pp=delta.min(),max_pp=delta.max(),negative_folds=int((delta<0).sum()),stability_lo=np.quantile(boot,.025),stability_hi=np.quantile(boot,.975)))
pd.DataFrame(effects).to_csv(R/'results/paired_effects.csv',index=False)
labels={'marginal':'Marginals','joint_copula':'Joint copula','conditional_copula':'Conditional copula'}
with (R/'paper/utility_table.tex').open('w') as f:
 for d in ['adult','bank','wine']:
  for g in labels:
   t=m[(m.dataset==d)&(m.generator==g)].groupby('arm').f1.mean()*100
   f.write(f'{d.title()} & {labels[g]} & '+ ' & '.join(f'{t[a]:.1f}' for a in ['raw','project','reject','quota'])+' '+chr(92)*2+chr(10))
with (R/'paper/audit_table.tex').open('w') as f:
 for d in ['adult','bank','wine']:
  t=m[(m.dataset==d)&(m.generator=='conditional_copula')].groupby('arm').mean(numeric_only=True)
  for a in ['raw','project','reject','quota']:
   v=t.loc[a];f.write(f'{d.title()} & {a.title()} & {v.violation_rate*100:.1f} & {v.prevalence_drift*100:.2f} & {v.candidate_ratio:.2f} & {v.marginal_distance:.3f} '+chr(92)*2+chr(10))
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(4.8,2.8),sharey=True)
for ax,d in zip(axes,['adult','bank','wine']):
 t=m[(m.dataset==d)&(m.generator=='conditional_copula')]
 for j,a in enumerate(['raw','project','reject','quota']):
  vals=t[t.arm==a].sort_values('fold').f1*100;ax.scatter(np.full(5,j)+np.linspace(-.1,.1,5),vals,s=13,color=['#555555','#0072B2','#D55E00','#009E73'][j]);ax.plot([j-.2,j+.2],[vals.mean()]*2,color='black',lw=1.4)
 ref=m[(m.dataset==d)&(m.generator=='real')].f1.mean()*100;ax.axhline(ref,color='black',ls='--',lw=.9)
 ax.set_title(d.title());ax.set_xticks(range(4),['Raw','Project','Reject','Quota'],rotation=55,ha='right');ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Positive-class F1 (%)');fig.text(.5,.015,'Dots: five folds; bars: means; dashed: real-training mean',ha='center',fontsize=8);fig.tight_layout(rect=(0,.07,1,1));fig.savefig(R/'paper/figures/utility.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(4.8,2.6))
for d,marker,color in [('adult','o','#0072B2'),('bank','s','#D55E00'),('wine','^','#009E73')]:
 t=b[(b.dataset==d)&(b.generator=='conditional_copula')];axes[0].scatter(t.acceptance_negative*100,t.acceptance_positive*100,label=d.title(),s=22,marker=marker,color=color)
 t=m[(m.dataset==d)&(m.generator=='conditional_copula')];p=t.pivot(index='fold',columns='arm',values='positive_fraction');axes[1].scatter((p['reject']-p['raw'])*100,(p['quota']-p['raw'])*100,s=22,marker=marker,color=color)
axes[0].plot([0,100],[0,100],ls='--',color='gray');axes[0].set(xlabel='Negative-class acceptance (%)',ylabel='Positive-class acceptance (%)',xlim=(0,104),ylim=(0,104))
axes[1].axhline(0,color='gray',lw=.7);axes[1].axvline(0,color='gray',lw=.7);axes[1].set(xlabel='Reject - raw (pp)',ylabel='Quota - raw (pp)')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False);fig.tight_layout(rect=(0,0,1,.88));fig.savefig(R/'paper/figures/selection.pdf');plt.close(fig)
print(pd.DataFrame(effects).to_string(index=False));print(b.groupby(['dataset','generator'])[['pool_positive','acceptance_negative','acceptance_positive','valid_pool_positive']].mean().round(4).to_string())
