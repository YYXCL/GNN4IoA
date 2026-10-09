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
    out.mkdir(parents=True,exist_ok=True);foundation=Foundation(ROOT/"1.数据")
    conn=sqlite3.connect(out/"samples.sqlite")
    conn.execute("create table if not exists samples(split text,id integer,data blob,primary key(split,id))")
    for split,count in zip(("train","val","test"),counts):
        existing={r[0] for r in conn.execute("select id from samples where split=?",(split,))}
        for i in range(count):
            if i in existing:continue
            e=foundation.episode(split,i,20260920,agents=100,difficulty=DIFFICULTIES[i%3])[0]
            e.snapshot.states={g:np.asarray(x,dtype=np.float32) for g,x in e.snapshot.states.items()}
            e.snapshot.missing={g:np.asarray(x,dtype=bool) for g,x in e.snapshot.missing.items()}
            conn.execute("insert into samples values(?,?,?)",(split,i,zlib.compress(pickle.dumps(e,protocol=5),1)))
            if (i+1)%100==0:conn.commit();print(f"generate {split}: {i+1}/{count}",flush=True)
        conn.commit();print(f"generate {split}: {count}/{count}",flush=True)
    conn.close()


def conditions():
    return ([('complexity',i,{'difficulty':d,'agents':100}) for i,d in enumerate(DIFFICULTIES)]
            +[('scale',n,{'difficulty':'中等','agents':n}) for n in [50,100,200,400,800,1600]]
            +[('concurrency',n,{'difficulty':'中等','agents':100,'batch':n}) for n in [1,2,4,8,16,32,64]])


def evaluate(out,seeds,instances,device,seconds,candidates=8,planning_batch_size=4):
    foundation=Foundation(ROOT/"1.数据")
    for seed in seeds:
        for method in METHODS:
            slug=method.lower().replace('-','_')
            model=load_model(out/"训练"/f"seed_{seed}"/slug/"best.pt",device)
            model.config.candidate_budgets=(candidates,)
            model.config.planning_batch_size=planning_batch_size
            for block,x,kwargs in conditions():
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['generate','train','evaluate','report','all'])
    parser.add_argument('--output',type=Path,default=ROOT/'4.结果输出'/'2026-10-09_TC-MDSM实验')
    parser.add_argument('--counts',default='12000,1500,1500')
    parser.add_argument('--seeds',default='7,17,29,43,61')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--instances',type=int,default=30)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--candidates',type=int,default=8)
    parser.add_argument('--planning-batch-size',type=int,default=4)
    parser.add_argument('--validation-limit',type=int,default=0)
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    counts=list(map(int,args.counts.split(',')));seeds=list(map(int,args.seeds.split(',')))
    if len(counts)!=3 or min(counts)<=0 or not seeds or min(args.epochs,args.instances,args.seconds,args.candidates,args.planning_batch_size)<=0:
        parser.error('counts 必须包含三个正数，种子非空，轮数、实例数和规划时间须为正')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    config=ModelConfig(candidate_budgets=(args.candidates,),planning_batch_size=args.planning_batch_size)
    if args.stage in ('generate','all'):
        save_json(out/'实验参数.json',{'counts':counts,'seeds':seeds,'epochs':args.epochs,'instances':args.instances,
                  'seconds':args.seconds,'validation_limit':args.validation_limit,'device':args.device,
                  'model':asdict(config),'methods':list(METHODS)})
        generate(out/'数据',counts)
    if args.stage in ('train','all'):
        train_cases=Cases(out/'数据'/'samples.sqlite','train');val_cases=Cases(out/'数据'/'samples.sqlite','val')
        for seed in seeds:
            for method in ([args.method] if args.method else METHODS):
                train(train_cases,val_cases,out/'训练'/f'seed_{seed}'/method.lower().replace('-','_'),method,
                      config,seed=seed,epochs=args.epochs,device=args.device,resume=args.resume,
                      validation_limit=args.validation_limit,time_limit=args.seconds)
    if args.stage in ('evaluate','all'):
        evaluate(out,seeds,args.instances,args.device,args.seconds,args.candidates,args.planning_batch_size)
    if args.stage in ('report','all'):
        from tc_mdsm_v3_report import report
        report(out)


if __name__=='__main__':main()
