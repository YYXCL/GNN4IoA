"""可断点续跑的 v3 全流程脚本：数据集生成、训练、冻结评估与汇总出图。
从仓库的工具目录下运行。训练与评估都不使用任何结果排名标签（不泄露答案）。
"""
from pathlib import Path
import argparse,collections,hashlib,json,pickle,sqlite3,sys,time,zlib
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1]    # 仓库根目录（本文件上一级的上一级）
sys.path.insert(0,str(ROOT/'2.模型'))       # 把模型包目录加入搜索路径，便于 import tc_mdsm
import numpy as np
import torch
from tc_mdsm.data import Foundation,DIFFICULTIES,seed_of
from tc_mdsm.types import METHODS,ModelConfig
from tc_mdsm.training import train,load_model,atomic_json
from tc_mdsm.execution import execute_batch
from tc_mdsm.provenance import code_identity,digest,model_identity

class Cases:
    """对 sqlite 数据集的一个只读、可索引（可切片/可迭代）的封装。"""
    def __init__(self,path,split):
        self.conn=sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True)  # 只读打开，避免误改数据集
        self.split=split
        self.count=self.conn.execute('select count(*) from samples where split=?',(split,)).fetchone()[0]
    def __len__(self):return self.count
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self))) ]  # 支持切片
        if index<0:index+=len(self)  # 支持负索引
        if not 0<=index<len(self):raise IndexError(index)
        row=self.conn.execute('select data from samples where split=? and id=?',(self.split,index)).fetchone()
        return pickle.loads(zlib.decompress(row[0]))  # 存的是 zlib 压缩后的 pickle
    def __iter__(self):
        for i in range(len(self)):yield self[i]

def generate(out,counts):
    # 生成数据集：把 Foundation 产生的 episode 落库为 sqlite，并按 split 分批提交、支持续跑
    out.mkdir(parents=True,exist_ok=True);f=Foundation(ROOT/'1.数据')
    # identity：数据集生成身份，记录 schema、种子、样本数、来源文件哈希与代码哈希，防止混用不同版本
    identity={'schema':'tc-mdsm/3','seed':20260920,'counts':counts,'agents':100,'sources':f.source_hashes,'code':code_identity()}
    marker=out/'generation_identity.json'
    def generation_signature(value):
        # 只取影响数据生成的三个源文件哈希作为签名，其余代码变化不强制重建数据集
        v=dict(value);v['code']={k:h for k,h in v['code'].items() if k in ('data.py','types.py','world.py')};return v
    if marker.exists() and generation_signature(json.loads(marker.read_text(encoding='utf-8')))!=generation_signature(identity):raise ValueError('dataset generation identity changed; use a new directory')
    atomic_json(marker,identity)
    conn=sqlite3.connect(out/'samples.sqlite');conn.execute('create table if not exists samples(split text,id integer,data blob,primary key(split,id))')
    for split,count in zip(('train','val','test'),counts):
        existing={r[0] for r in conn.execute('select id from samples where split=?',(split,))}  # 已存在的样本跳过，实现续跑
        for i in range(count):
            if i in existing:continue
            e=f.episode(split,i,20260920,agents=100,difficulty=DIFFICULTIES[i%3])[0]
            # 把快照状态转成紧凑的 numpy 数组，减小体积
            e.snapshot.states={g:np.asarray(x,dtype=np.float32) for g,x in e.snapshot.states.items()}
            e.snapshot.missing={g:np.asarray(x,dtype=bool) for g,x in e.snapshot.missing.items()}
            conn.execute('insert into samples values(?,?,?)',(split,i,zlib.compress(pickle.dumps(e,protocol=5),1)))
            if (i+1)%100==0:conn.commit();print(json.dumps({'stage':'generate','split':split,'done':i+1,'total':count}),flush=True)  # 每 100 条提交并打印进度
        conn.commit()
    conn.close()
    # 生成完成后做数据审计：候选数量分布、标签分布、模板/时间是否与 split 隔离
    audit={};templates={};bounds={}
    for split in ('train','val','test'):
        counts_=[];labels=collections.Counter();groups=set();templates[split]=set();times=[]
        for e in Cases(out/'samples.sqlite',split):
            e.snapshot.validate();e.workflow.validate()  # 结构合法性校验
            counts_ += [len(e.snapshot.feasible(t)) for t in e.workflow.tasks]  # 每个任务的可选智能体数量
            labels.update((o.kind,str(o.success)) for o in e.observations)
            groups.add(e.provenance['trace_group']);templates[split].add(e.workflow.template)
            times.extend([e.provenance['trace_start'],e.provenance['trace_end']])
        bounds[split]=[min(times),max(times)]
        audit[split]={'workflows':len(Cases(out/'samples.sqlite',split)),'candidate_mean':float(np.mean(counts_)),'candidate_p50':float(np.median(counts_)),'fraction_over32':float(np.mean(np.array(counts_)>32)),'zero_fraction':float(np.mean(np.array(counts_)==0)),'labels':{str(k):v for k,v in labels.items()},'trace_groups':len(groups),'time_bounds':bounds[split]}
    # 断言模板不跨 split 重叠、时间窗口不交叉，避免数据泄露
    assert not templates['train']&templates['val'] and not templates['train']&templates['test'] and not templates['val']&templates['test']
    assert bounds['train'][1]<bounds['val'][0] and bounds['val'][1]<bounds['test'][0]
    atomic_json(out/'audit.json',audit)
    # 计算 sqlite 文件的 sha256，写入 manifest，后续训练前会校验一致性
    h=hashlib.sha256()
    with (out/'samples.sqlite').open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    atomic_json(out/'manifest.json',{**identity,'sqlite_sha256':h.hexdigest(),'audit':audit})

def conditions():
    # 全部评估条件：复杂度 3 档 + 规模 6 档 + 并发 7 档，共 16 档（与 report 中的 16 对应）
    return [('complexity',i,{'difficulty':d,'agents':100}) for i,d in enumerate(DIFFICULTIES)]+[('scale',n,{'difficulty':'中等','agents':n}) for n in [50,100,200,400,800,1600]]+[('concurrency',n,{'difficulty':'中等','agents':100,'batch':n}) for n in [1,2,4,8,16,32,64]]

def evaluate(out,instances,device,seconds):
    # 冻结评估：在冻结协议下，对每个种子/方法/条件/实例跑仿真，结果逐文件落盘（已存在则跳过，支持续跑）
    freeze=json.loads((out/'冻结协议.json').read_text(encoding='utf-8'))
    if freeze['code']!=code_identity():raise ValueError('source changed since freeze')       # 代码变了就不允许继续评估
    f=Foundation(ROOT/'1.数据')
    if freeze['sources']!=f.source_hashes:raise ValueError('capability sources changed since freeze')  # 输入源变了也不允许
    for seed in freeze['seeds']:
        for method in METHODS:
            slug=method.lower().replace('-','_');checkpoint=out/'训练'/f'seed_{seed}'/slug/'best.pt'
            # checkpoint 哈希必须与冻结时记录一致，防止中途换模型
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=freeze['checkpoints'][f'{seed}/{method}']:raise ValueError('checkpoint changed')
            model=load_model(checkpoint,device)
            for block,x,kwargs in conditions():
                for i in range(instances):
                    dest=out/'逐实例结果'/f'seed_{seed}'/slug/f'{block}_{x}_{i}.json'
                    if dest.exists():continue  # 已有结果直接跳过
                    idx=seed_of('formal-v3',block,x,i)%100000000  # 由条件派生一个稳定的实例索引
                    examples=f.episode('test',idx,seed,**kwargs)
                    with torch.no_grad():rows,events,plans=execute_batch(model,examples,time_limit=seconds,seed=seed)  # 跑一次仿真执行
                    for row in rows:row.update(seed=seed,method=method,block=block,x=x,instance=i,trace_group=examples[0].provenance['trace_group'])  # 给每条结果补上分组信息
                    atomic_json(dest,{'rows':rows,'executions':events,'plans':[asdict(p) for p in plans],'freeze_digest':digest(freeze)})
                    print(json.dumps({'stage':'evaluate','seed':seed,'method':method,'block':block,'x':x,'instance':i,'completed':sum(r['on_time'] for r in rows),'requests':len(rows)}),flush=True)

def main():
    # 命令行入口：stage 决定执行哪个阶段；all 表示从头跑完整流程
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['generate','train','freeze','evaluate','report','all'])
    parser.add_argument('--output',type=Path,default=ROOT/'6.结果输出'/'2026-09-27_TC-MDSM正式实验'/'运行')  # 输出目录
    parser.add_argument('--counts',default='12000,1500,1500');parser.add_argument('--seeds',default='7,17,29,43,61')  # 各 split 样本数、随机种子
    parser.add_argument('--epochs',type=int,default=100);parser.add_argument('--instances',type=int,default=30)      # 训练轮数、每条件实例数
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');parser.add_argument('--seconds',type=float,default=30)  # 设备、单次决策时间上限
    parser.add_argument('--validation-limit',type=int,default=0);parser.add_argument('--method',choices=METHODS)   # 验证集截断、只跑单个方法
    args=parser.parse_args();torch.set_num_threads(2);out=args.output;out.mkdir(parents=True,exist_ok=True)
    counts=list(map(int,args.counts.split(',')));seeds=list(map(int,args.seeds.split(',')))
    if args.stage in ('generate','all'):generate(out/'数据',counts)
    if args.stage in ('train','all'):
        dataset=out/'数据'
        # 训练前先校验数据集未被改动（sha256 与 manifest 一致）
        manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'));h=hashlib.sha256()
        with (dataset/'samples.sqlite').open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=manifest['sqlite_sha256']:raise ValueError('dataset checksum mismatch')
        train_cases=Cases(dataset/'samples.sqlite','train');val_cases=Cases(dataset/'samples.sqlite','val')
        for seed in seeds:
            for method in ([args.method] if args.method else METHODS):  # 指定了 --method 就只训该方法
                print(json.dumps({'stage':'train_start','seed':seed,'method':method,'device':args.device}),flush=True)
                train(dataset,out/'训练'/f'seed_{seed}'/method.lower().replace('-','_'),method,ModelConfig(),seed=seed,epochs=args.epochs,patience=10,device=args.device,resume=True,validation_limit=args.validation_limit,time_limit=args.seconds,train_cases=train_cases,val_cases=val_cases)
    if args.stage in ('freeze','all'):
        # 冻结阶段：校验训练确实达到了停止条件，然后把所有 hash 写进冻结协议
        checkpoints={}
        for seed in seeds:
            for method in METHODS:
                ck=out/'训练'/f'seed_{seed}'/method.lower().replace('-','_')/'best.pt'
                if method!='Rule-Latency':  # 规则基线无需训练，跳过训练状态校验
                    last=torch.load(ck.with_name('last.pt'),map_location='cpu',weights_only=False)
                    run=json.loads(ck.with_name('run_manifest.json').read_text(encoding='utf-8'))
                    if run['code']!=code_identity():raise ValueError('training code differs from freezing code')  # 训练与冻结的代码必须一致
                    if not (last['epoch']>=args.epochs or (last['epoch']>=7 and last['stale']>=10)):raise ValueError('training has not reached its configured stopping condition')  # 未正常收敛/早停则拒绝冻结
                    if run['validation_limit']!=args.validation_limit or run['solver_seconds']!=args.seconds:raise ValueError('training validation protocol differs')  # 训练与冻结的评估口径要一致
                checkpoints[f'{seed}/{method}']=hashlib.sha256(ck.read_bytes()).hexdigest()
        # formal 标记：只有用正式配置（默认样本数、种子、实例数、轮数、时间上限）才认定为正式实验
        freeze={'seeds':seeds,'instances':args.instances,'epochs':args.epochs,'counts':counts,'hidden':128,'candidate_budget':32,'seconds':args.seconds,'code':code_identity(),'sources':Foundation(ROOT/'1.数据').source_hashes,'checkpoints':checkpoints,'formal':counts==[12000,1500,1500] and seeds==[7,17,29,43,61] and args.instances==30 and args.epochs==50 and args.validation_limit==0 and args.seconds==30}
        dest=out/'冻结协议.json'
        if dest.exists() and json.loads(dest.read_text(encoding='utf-8'))!=freeze:raise ValueError('existing protocol differs')  # 已存在且不一致则报错
        atomic_json(dest,freeze)
    if args.stage in ('evaluate','all'):evaluate(out,args.instances,args.device,args.seconds)
    if args.stage in ('report','all'):
        from tc_mdsm_v3_report import report  # 延迟导入，报告阶段才需要 matplotlib
        report(out)
if __name__=='__main__':main()
