"""Pooled, seed-level reporting; incomplete runs cannot masquerade as formal results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t,ttest_1samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tc_mdsm.types import METHODS

def report(out):
    out=Path(out);freeze=json.loads((out/'冻结协议.json').read_text(encoding='utf-8'))
    records=[];decisions=[]
    for file in sorted((out/'逐实例结果').rglob('*.json')):
        d=json.loads(file.read_text(encoding='utf-8'));records.extend(d['rows'])
        r=d['rows'][0];p=d['plans'][0]
        decisions.append({k:r[k] for k in ['seed','method','block','x','instance']}|{'decision_s':p['elapsed'],'status':p['status'],'accepted':sum(p['admitted'].values()),'requests':len(d['rows'])})
    expected=len(freeze['seeds'])*len(METHODS)*16*freeze['instances']
    if len(decisions)!=expected:raise ValueError(f'incomplete matrix: {len(decisions)}/{expected}; no final plots produced')
    conditions=[('complexity',x) for x in range(3)]
    conditions += [('scale',x) for x in (50,100,200,400,800,1600)]
    conditions += [('concurrency',x) for x in (1,2,4,8,16,32,64)]
    actual={(d['seed'],d['method'],d['block'],d['x'],d['instance']) for d in decisions}
    required={(seed,method,block,x,i) for seed in freeze['seeds'] for method in METHODS
              for block,x in conditions for i in range(freeze['instances'])}
    if actual!=required:
        raise ValueError(f'evaluation matrix keys differ from protocol: {len(required-actual)} missing, {len(actual-required)} unexpected')
    folder=out/'实验结果图';folder.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(records);batch=pd.DataFrame(decisions)
    if set(df.method)!=set(METHODS):
        raise ValueError(f'run methods {sorted(df.method.unique())} do not match current protocol {METHODS}')
    df.to_csv(folder/'逐请求结果.csv',index=False,encoding='utf-8-sig');batch.to_csv(folder/'逐批决策耗时.csv',index=False,encoding='utf-8-sig')
    decision=batch.groupby(['block','x','method']).decision_s.agg(mean='mean',p50='median',p95=lambda x:x.quantile(.95),batches='count').reset_index()
    decision.to_csv(folder/'决策时延统计.csv',index=False,encoding='utf-8-sig')
    def ci(values):
        a=np.array(values,dtype=float);a=a[np.isfinite(a)]
        if not len(a):return np.nan,np.nan
        return a.mean(),t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0.
    stats=[]
    for (block,x,method),g in df.groupby(['block','x','method']):
        first=g.groupby(['seed','instance']).on_time.mean().groupby('seed').mean()
        latency=g[g.on_time==1].groupby('seed').completion_s.mean()
        rate,width=ci(first);seconds,error=ci(latency)
        stats.append(dict(block=block,x=x,method=method,completion_rate=rate,completion_rate_ci=width,latency_s=seconds,latency_ci=error,latency_seeds=len(latency),successes=int(g.on_time.sum()),requests=len(g),penalized_s=g.groupby('seed').penalized_s.mean().mean()))
    summary=pd.DataFrame(stats);summary.to_csv(folder/'主结果统计.csv',index=False,encoding='utf-8-sig')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    methods=list(METHODS)
    metrics=[('completion_rate','completion_rate_ci','任务完成率'),
             ('latency_s','latency_ci','端到端平均完成时延（秒）')]
    def save(fig,name,top=1.):
        fig.tight_layout(rect=(0,0,1,top))
        for ext in ('png','pdf'):
            fig.savefig(folder/f'{name}.{ext}',dpi=220)
        plt.close(fig)
    complexity=summary[summary.block=='complexity']
    xs=sorted(complexity.x.unique())
    if len(xs)!=3:raise ValueError('complexity experiment requires three difficulty levels')
    for number,(metric,error,ylabel) in enumerate(metrics,1):
        fig,ax=plt.subplots(figsize=(8.5,4.8))
        width=.13
        for j,method in enumerate(methods):
            g=complexity[complexity.method==method].set_index('x').reindex(xs)
            y=g[metric].to_numpy(dtype=float)
            e=np.nan_to_num(g[error].to_numpy(dtype=float),nan=0.)
            ax.bar(np.arange(len(xs))+(j-(len(methods)-1)/2)*width,y,width,
                   yerr=e,capsize=2,label='Ours (TC-MDSM)' if method=='TC-MDSM' else method)
        ax.set_xticks(np.arange(len(xs)),['简单','中等','困难'])
        ax.set_xlabel('任务复杂度');ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
        ax.legend(ncol=3,fontsize=8)
        if metric=='completion_rate':ax.set_ylim(0,1.02)
        if not freeze['formal']:ax.set_title('贯通验证：非正式实验结果',color='darkred')
        save(fig,f'{number}.{ "任务完成率" if number==1 else "任务平均完成时延" }')
    for number,block,xlabel,name in ((3,'concurrency','并行任务数量','并行场景的模型性能'),
                                      (4,'scale','智能体数量','模型随系统规模扩大的性能变化')):
        sub=summary[summary.block==block]
        xs=sorted(sub.x.unique())
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for ax,(metric,error,ylabel) in zip(axes,metrics):
            for method in methods:
                g=sub[sub.method==method].set_index('x').reindex(xs)
                ax.errorbar(xs,g[metric].to_numpy(dtype=float),
                            yerr=np.nan_to_num(g[error].to_numpy(dtype=float),nan=0.),
                            marker='o',capsize=2,linewidth=2 if method=='TC-MDSM' else 1.3,
                            label='Ours (TC-MDSM)' if method=='TC-MDSM' else method)
            ax.set(xlabel=xlabel,ylabel=ylabel,xscale='log',xticks=xs)
            ax.set_xticklabels([str(x) for x in xs]);ax.grid(alpha=.2)
            if metric=='completion_rate':ax.set_ylim(0,1.02)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.97),ncol=3,fontsize=8)
        if not freeze['formal']:fig.suptitle('贯通验证：非正式实验结果',color='darkred',y=1.02)
        save(fig,f'{number}.{name}',top=.81)
    paired=[];common=[]
    for (block,x),g in df.groupby(['block','x']):
        for method in methods[1:]:
            for metric in ['on_time','penalized_s']:
                means=g.groupby(['method','seed', 'instance'])[metric].mean().groupby(['method','seed']).mean().unstack(0)
                delta=(means['TC-MDSM']-means[method]).dropna();mean,width=ci(delta)
                p=float(ttest_1samp(delta,0).pvalue) if len(delta)>1 and delta.std()>0 else (1. if mean==0 else None)
                paired.append(dict(block=block,x=x,baseline=method,metric=metric,difference=mean,ci_low=mean-width,ci_high=mean+width,p=p))
            keys=['seed','instance','workflow'];a=g[(g.method=='TC-MDSM')&(g.on_time==1)];b=g[(g.method==method)&(g.on_time==1)]
            merged=a.merge(b,on=keys,suffixes=('_tc','_baseline'))
            delta=(merged.completion_s_tc-merged.completion_s_baseline).groupby(merged.seed).mean();mean,width=ci(delta)
            common.append(dict(block=block,x=x,baseline=method,common_successes=len(merged),difference_s=mean,ci_low=mean-width,ci_high=mean+width))
    comparison=pd.DataFrame(paired)
    for key,idx in comparison.groupby(['block','x','metric']).groups.items():
        ordered=sorted(idx,key=lambda i:comparison.loc[i,'p'] if pd.notna(comparison.loc[i,'p']) else 1.)
        maximum=0
        for rank,i in enumerate(ordered):
            p=comparison.loc[i,'p'];maximum=max(maximum,min(1.,(len(ordered)-rank)*p)) if pd.notna(p) else 1.
            comparison.loc[i,'holm_p']=maximum
    comparison.to_csv(folder/'种子配对差值.csv',index=False,encoding='utf-8-sig');pd.DataFrame(common).to_csv(folder/'共同成功请求时延.csv',index=False,encoding='utf-8-sig')
    (folder/'结果说明.md').write_text(f'# 实验结果\n\n正式协议：{freeze["formal"]}。完整实例文件：{len(decisions)}。\n\n完成率以复杂请求为单位，首次异常、拒绝和超时均计失败。时延只统计首次按期成功请求，未成功不记零。误差区间以种子为单位；成功样本数及有效时延种子数见主结果统计。并发决策耗时按批统计。\n\n所有数值来自逐事件仿真，不是实测智能体部署结果。\n',encoding='utf-8')
    print(str(folder),flush=True)
