"""汇总完成率和成功请求时延，输出四幅实验图。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')  # 使用无界面的 Agg 后端，方便在服务器/脚本中直接出图
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from tc_mdsm.types import METHODS

METHOD_LABELS = {"HM-MATAS": "TARGETNET"}

def report(out, paper_style=False):
    out=Path(out);parameters=json.loads((out/'实验参数.json').read_text(encoding='utf-8'))
    records=[];decisions=[]  # records：逐请求（workflow）明细；decisions：逐批（一次决策）的耗时与状态
    for file in sorted((out/'逐实例结果').rglob('*.json')):
        d=json.loads(file.read_text(encoding='utf-8'));records.extend(d['rows'])
        r=d['rows'][0];p=d['plans'][0]  # 每个实例文件的行共享同一 seed/method/block/x/instance，故取第一行即可
        decisions.append({k:r[k] for k in ['seed','method','block','x','instance']}|{'decision_s':p['elapsed'],'status':p['status'],'accepted':sum(p['admitted'].values()),'requests':len(d['rows'])}|p.get('timings',{}))
    if not records:raise ValueError('没有可绘制的评估结果')
    folder=out/'实验结果图';folder.mkdir(parents=True,exist_ok=True)  # 所有图表和 CSV 统一输出到这里
    df=pd.DataFrame(records);batch=pd.DataFrame(decisions)
    # 导出明细：逐请求结果、逐批决策耗时
    df.to_csv(folder/'逐请求结果.csv',index=False,encoding='utf-8-sig');batch.to_csv(folder/'逐批决策耗时.csv',index=False,encoding='utf-8-sig')
    # 按（条件块, 取值, 方法）汇总决策时延：均值、中位数、95 分位、批次数
    decision=batch.groupby(['block','x','method']).decision_s.agg(mean='mean',p50='median',p95=lambda x:x.quantile(.95),batches='count').reset_index()
    decision.to_csv(folder/'决策时延统计.csv',index=False,encoding='utf-8-sig')
    def ci(values):
        # 计算均值及其 95% 置信区间半宽（基于 t 分布，以种子为独立统计单位）
        a=np.array(values,dtype=float);a=a[np.isfinite(a)]  # 过滤掉 NaN/inf
        if not len(a):return np.nan,np.nan
        return a.mean(),t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else np.nan
    stats=[]
    for (block,x,method),g in df.groupby(['block','x','method']):
        # 完成率：先在组内对实例求均值，再对种子求均值（每颗种子权重相同）
        first=g.groupby(['seed','instance']).on_time.mean().groupby('seed').mean()
        # 时延：只统计按期成功（on_time==1）的请求；失败不记 0，直接排除
        latency=g[g.on_time==1].groupby('seed').completion_s.mean()
        rate,width=ci(first);seconds,error=ci(latency)
        stats.append(dict(block=block,x=x,method=method,completion_rate=rate,completion_rate_ci=width,
                          admitted_rate=g.groupby('seed').initial_admitted.mean().mean(),
                          executed_completion_rate=g.groupby('seed').completed.mean().mean(),
                          latency_s=seconds,latency_ci=error,latency_seeds=len(latency),successes=int(g.on_time.sum()),
                          requests=len(g),penalized_s=g.groupby('seed').penalized_s.mean().mean()))
    summary=pd.DataFrame(stats);summary.to_csv(folder/'主结果统计.csv',index=False,encoding='utf-8-sig')
    # 中文字体设置，保证图中中文正常显示
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.prop_cycle':plt.cycler(color=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9'])})
    if paper_style:
        plt.rcParams.update({'font.family':'DejaVu Sans','font.sans-serif':['DejaVu Sans']})
    methods=parameters['methods']
    markers=['o','s','^','D','v','P']
    # 两个核心指标：完成率、端到端平均完成时延
    metrics=[('completion_rate','completion_rate_ci','按期任务完成率'),
             ('latency_s','latency_ci','端到端平均完成时延（秒）')]
    if paper_style:
        metrics=[('completion_rate','completion_rate_ci','On-time completion rate'),
                 ('latency_s','latency_ci','Average completion latency (s)')]
    def save(fig,name,top=1.):
        # 统一保存为 png 与 pdf 两种格式
        fig.tight_layout(rect=(0,0,1,top))
        for ext in ('png','pdf'):
            fig.savefig(folder/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    # ===== 实验一/二：任务复杂度对比（柱状图，3 档难度） =====
    complexity=summary[summary.block=='complexity']
    xs=sorted(complexity.x.unique())
    # 每个指标画一张分组柱状图：每个难度下并排展示各方法
    for number,(metric,error,ylabel) in enumerate(metrics,1):
        fig,ax=plt.subplots(figsize=(8.5,4.8))
        width=.13
        for j,method in enumerate(methods):
            g=complexity[complexity.method==method].set_index('x').reindex(xs)
            y=g[metric].to_numpy(dtype=float)
            e=np.nan_to_num(g[error].to_numpy(dtype=float),nan=0.)
            # 仅限制误差条的显示范围，CSV保留原始区间。
            e=np.vstack([np.minimum(e,np.nan_to_num(y)),
                         np.minimum(e,np.nan_to_num(1-y)) if metric=='completion_rate' else e])
            errors={} if paper_style else dict(yerr=e,capsize=2,error_kw={'elinewidth':1.1,'capthick':1.1})
            ax.bar(np.arange(len(xs))+(j-(len(methods)-1)/2)*width,y,width,**errors,
                   edgecolor='white',linewidth=.6,label=METHOD_LABELS.get(method,method))
        ax.set_xticks(np.arange(len(xs)),(['Easy','Medium','Hard'] if paper_style else ['简单','中等','困难']))
        ax.set_xlabel('Task complexity' if paper_style else '任务复杂度');ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(ncol=3,fontsize=9,loc='upper center',bbox_to_anchor=(.5,1.20),frameon=False)
        if metric=='completion_rate':ax.set_ylim(0,1.02)
        else:ax.set_ylim(bottom=0)
        save(fig,f'{number}.{ "任务完成率" if number==1 else "任务平均完成时延" }')
    # ===== 实验三/四：并发规模 / 系统规模（双栏折线图，横轴取对数） =====
    for number,block,xlabel,name in ((3,'concurrency','并行任务数量','并行场景的模型性能'),
                                      (4,'scale','智能体数量','模型随系统规模扩大的性能变化')):
        if paper_style:
            xlabel='Concurrent workflows' if block=='concurrency' else 'Number of agents'
        sub=summary[summary.block==block]
        xs=sorted(sub.x.unique())
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for ax,(metric,error,ylabel) in zip(axes,metrics):  # 左右两栏分别为完成率与时延
            for j,method in enumerate(methods):
                g=sub[sub.method==method].set_index('x').reindex(xs)
                y=g[metric].to_numpy(dtype=float)
                e=np.nan_to_num(g[error].to_numpy(dtype=float),nan=0.)
                e=np.vstack([np.minimum(e,np.nan_to_num(y)),
                             np.minimum(e,np.nan_to_num(1-y)) if metric=='completion_rate' else e])
                line=dict(marker=markers[j],markersize=5,linewidth=2 if method=='TC-MDSM' else 1.3,label=METHOD_LABELS.get(method,method))
                if paper_style:ax.plot(xs,y,**line)
                else:ax.errorbar(xs,y,yerr=e,capsize=2,elinewidth=1,**line)
            ax.set(xlabel=xlabel,ylabel=ylabel)
            ax.set_xscale('log',base=2 if paper_style else 10);ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs]);ax.xaxis.set_minor_formatter(NullFormatter());ax.grid(alpha=.2)
            if paper_style:
                ax.minorticks_off();ax.set_axisbelow(True)
                ax.grid(axis='x',visible=False)
            if metric=='completion_rate':ax.set_ylim(0,1.02)
            else:ax.set_ylim(bottom=0)
            if block=='scale' and metric=='latency_s' and not paper_style:
                # 少量种子的区间很宽，压缩高值以同时看清均值与完整上界。
                ax.set_yscale('symlog',linthresh=20)
                upper=ax.get_ylim()[1]
                ticks=[value for value in [0,20,50,100,200,500,1000] if value<=upper]
                ax.set_yticks(ticks)
                ax.set_yticklabels([str(value) for value in ticks])
                ax.set_ylabel('端到端平均完成时延（秒，非线性刻度）')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.97),ncol=3,fontsize=9,frameon=False)
        save(fig,f'{number}.{name}',top=.81)

    display_note=(
        '图中使用英文标注，分别输出两幅分组柱状图和两幅双栏折线图，每幅有独立图例，不显示误差条或阴影。'
        'CSV保留种子均值的95% t区间；不显示区间不代表波动为零。折线横轴以2为底取对数，纵轴为线性刻度。\n\n'
        if paper_style else
        '误差条为种子均值的95%区间。完成率误差条只显示0到1范围，时延区间下界只显示非负部分，CSV保留原始t区间。'
        '第四图时延纵轴使用非线性刻度（symlog，20秒内线性），同时显示均值与较宽区间上界。\n\n')
    (folder/'结果说明.md').write_text(
        '# 实验结果\n\n'
        f"种子：{parameters['seeds']}；每条件每种子 {parameters['instances']} 个实例；训练样本 {parameters['counts'][0]} 条；最多 {parameters['epochs']} 轮；规划预算 {parameters['seconds']} 秒。\n\n"
        '完成率以整个工作流首次按期成功为准，拒绝、执行失败及迟到均计失败。时延只统计按期成功请求，零成功时留空，图中的缺失时延不是零时延。少于两个有效种子时不计算区间。少量种子和实例会产生较大波动。\n\n'
        +display_note+
        '图表来自动态仿真；本报告只汇总已有结果，运行规模见实验参数。\n',encoding='utf-8')
    print(str(folder),flush=True)
