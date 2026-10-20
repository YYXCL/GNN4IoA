"""实验入口：数据生成 → 训练 → 评估 → 绘图。"""
from pathlib import Path
import argparse
import json
import pickle
import sqlite3
import sys
import zlib
from dataclasses import asdict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"2.模型"))
import numpy as np
import torch
from tc_mdsm.data import Foundation, DIFFICULTIES, seed_of
from tc_mdsm.types import METHODS, ModelConfig
from tc_mdsm.training import train, load_model, save_json
from tc_mdsm.execution import execute_batch


class Cases:
    """按索引读取 SQLite 中的实验样本。"""
    def __init__(self,path,split):
        self.conn=sqlite3.connect(f"file:{path.as_posix()}?mode=ro",uri=True)
        self.split=split
        self.count=self.conn.execute("select count(*) from samples where split=?",(split,)).fetchone()[0]
    def __len__(self):return self.count
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        row=self.conn.execute("select data from samples where split=? and id=?",(self.split,index)).fetchone()
        return pickle.loads(zlib.decompress(row[0]))
    def __iter__(self):
        for i in range(len(self)):yield self[i]


def generate(out,counts):
    out.mkdir(parents=True,exist_ok=True);foundation=Foundation(ROOT/"1.数据")    # 确保输出目录存在。创建 Foundation 对象
    conn=sqlite3.connect(out/"samples.sqlite")      #在输出目录下打开/新建 samples.sqlite
    conn.execute("create table if not exists samples(split text,id integer,data blob,primary key(split,id))")
    for split,count in zip(("train","val","test"),counts):      # 按划分(12000,1500,1500)生成数据
        existing={r[0] for r in conn.execute("select id from samples where split=?",(split,))}      #
        for i in range(count):
            if i in existing:continue
            e=foundation.episode(split,i,20260920,agents=100,difficulty=DIFFICULTIES[i%3])[0]
            e.snapshot.states={g:np.asarray(x,dtype=np.float32) for g,x in e.snapshot.states.items()}   # 统一类型，把snapshot.states转成 numpy 数组（float32 / bool）
            e.snapshot.missing={g:np.asarray(x,dtype=bool) for g,x in e.snapshot.missing.items()}   # 统一类型，把snapshot.missing转成 numpy 数组（float32 / bool）
            conn.execute("insert into samples values(?,?,?)",(split,i,zlib.compress(pickle.dumps(e,protocol=5),1)))     # 插入数据库：pickle 序列化 → zlib.compress(..., 1) 压缩→ 存进 blob
            if (i+1)%100==0:conn.commit();print(f"generate {split}: {i+1}/{count}",flush=True)      # 每 100 条提交一次事务并打印进度。
        conn.commit();print(f"generate {split}: {count}/{count}",flush=True)
    conn.close()


def conditions(scale_agents=(50,100,200,400,800,1600),concurrency=(1,2,4,8,16,32,64)):
    return ([('complexity',i,{'difficulty':d,'agents':100}) for i,d in enumerate(DIFFICULTIES)]
            +[('scale',n,{'difficulty':'中等','agents':n}) for n in scale_agents]
            +[('concurrency',n,{'difficulty':'中等','agents':100,'batch':n}) for n in concurrency])


def evaluate(out,seeds,instances,device,seconds,candidates=8,planning_batch_size=4,methods=METHODS,
             scale_agents=(50,100,200,400,800,1600),concurrency=(1,2,4,8,16,32,64),deadline_policy="hard",solver_start="serial",
             execution_policy="ready"):
    foundation=Foundation(ROOT/"1.数据")
    for seed in seeds:
        for method in methods:
            slug=method.lower().replace('-','_')
            model=load_model(out/"训练"/f"seed_{seed}"/slug/"best.pt",device)
            model.config.candidate_budgets=(candidates,)
            model.config.planning_batch_size=planning_batch_size
            model.config.deadline_policy=deadline_policy
            model.config.solver_start=solver_start
            model.config.execution_policy=execution_policy
            for block,x,kwargs in conditions(scale_agents,concurrency):
                for i in range(instances):
                    dest=out/"逐实例结果"/f"seed_{seed}"/slug/f"{block}_{x}_{i}.json"
                    if dest.exists():continue
                    idx=seed_of('formal-v3',block,x,i)%100000000
                    examples=foundation.episode('test',idx,seed,**kwargs)
                    with torch.no_grad():rows,events,plans=execute_batch(model,examples,time_limit=seconds,seed=seed)
                    for row in rows:row.update(seed=seed,method=method,block=block,x=x,instance=i)
                    save_json(dest,{'rows':rows,'executions':events,'plans':[asdict(p) for p in plans]})
                    print(f"evaluate seed={seed} {method} {block}={x} {i+1}/{instances} on_time={sum(r['on_time'] for r in rows)}/{len(rows)}",flush=True)
                    print('timings '+json.dumps(plans[0].timings),flush=True)
                model._state_cache=None;model._index_cache=None
            del model
            if device.startswith('cuda'):torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)     # 创建一个参数解析器
    parser.add_argument('stage',choices=['generate','train','evaluate','report','all'])     # 决定脚本运行哪个阶段
    parser.add_argument('--output',type=Path,default=ROOT/'4.结果输出'/'2026-10-12_实验调试')    # 输出目录
    parser.add_argument('--counts',default='12000,1500,1500')       # 数据量配置
    parser.add_argument('--seeds',default='7,17,29,43,61')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--instances',type=int,default=30)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--candidates',type=int,default=8)
    parser.add_argument('--planning-batch-size',type=int,default=4)
    parser.add_argument('--validation-limit',type=int,default=0)
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--scale-agents',default='50,100,200,400,800,1600')
    parser.add_argument('--concurrency',default='1,2,4,8,16,32,64')
    parser.add_argument('--deadline-policy',choices=['hard','evaluate'],default='hard')
    parser.add_argument('--solver-start',choices=['serial','reject'],default='serial')
    parser.add_argument('--execution-policy',choices=['ready','planned'],default='ready')
    parser.add_argument('--samples',type=Path,help='复用已有样本数据库，避免对照实验复制相同数据')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)   # 真正读取命令行输入，返回一个 args 对象。把 PyTorch 的 CPU 线程数限制为 2。
    counts=list(map(int,args.counts.split(',')));seeds=list(map(int,args.seeds.split(',')))     # 命令行传进来的是字符串。这里把逗号分隔字符串转成整数列表
    scale_agents=list(map(int,args.scale_agents.split(',')));concurrency=list(map(int,args.concurrency.split(',')))
    methods=[args.method] if args.method else list(METHODS)
    if len(counts)!=3 or min(counts)<=0 or not seeds or min(args.epochs,args.instances,args.seconds,args.candidates,args.planning_batch_size)<=0:   # 对命令行的输入参数解析之后做合法性校验。
        parser.error('counts 必须包含三个正数，种子非空，轮数、实例数和规划时间须为正')
    if not scale_agents or not concurrency or min(scale_agents+concurrency)<=0:
        parser.error('系统规模和并发数必须为正')
    out=args.output;out.mkdir(parents=True,exist_ok=True)       # 创建输出目录
    config=ModelConfig(candidate_budgets=(args.candidates,),planning_batch_size=args.planning_batch_size,       # 把命令行参数打包成一个 ModelConfig 对象。
                       deadline_policy=args.deadline_policy,solver_start=args.solver_start,
                       execution_policy=args.execution_policy)
    if args.stage in ('generate','all'):        # 第一步数据生成：保存实验参数快照 + 按需生成样本数据。
        save_json(out/'实验参数.json',{'counts':counts,'seeds':seeds,'epochs':args.epochs,'instances':args.instances,
                  'seconds':args.seconds,'validation_limit':args.validation_limit,'device':args.device,
                  'model':asdict(config),'methods':methods,'scale_agents':scale_agents,'concurrency':concurrency,
                  'topology':'agent_complete_compute_leaf','duration_loss':'log_ratio',     # 新加的，感觉没啥必要
                  'samples_source':str(args.samples.resolve()) if args.samples else str((out/'数据'/'samples.sqlite').resolve())})
        if args.samples is None:generate(out/'数据',counts)
    if args.stage in ('train','all'):
        sample_path=args.samples or out/'数据'/'samples.sqlite'
        train_cases=Cases(sample_path,'train');val_cases=Cases(sample_path,'val')
        for seed in seeds:
            for method in ([args.method] if args.method else METHODS):
                train(train_cases,val_cases,out/'训练'/f'seed_{seed}'/method.lower().replace('-','_'),method,
                      config,seed=seed,epochs=args.epochs,device=args.device,resume=args.resume,
                      validation_limit=args.validation_limit,time_limit=args.seconds)
    if args.stage in ('evaluate','all'):
        evaluate(out,seeds,args.instances,args.device,args.seconds,args.candidates,args.planning_batch_size,
                 methods,scale_agents,concurrency,args.deadline_policy,args.solver_start,args.execution_policy)
    if args.stage in ('report','all'):
        from tc_mdsm_v3_report import report
        report(out)


if __name__=='__main__':main()
