"""Restartable v3 dataset, training, frozen evaluation and pooled figures.

Run from the repository's tools directory. No result-ranking targets are used.
"""
from pathlib import Path
import argparse,collections,hashlib,json,pickle,sqlite3,sys,time,zlib
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'2.模型'))
import numpy as np
import torch
from tc_mdsm.data import Foundation,DIFFICULTIES,seed_of
from tc_mdsm.types import METHODS,ModelConfig
from tc_mdsm.training import train,load_model,atomic_json
from tc_mdsm.execution import execute_batch
from tc_mdsm.provenance import code_identity,digest,model_identity

class Cases:
    def __init__(self,path,split):
        self.conn=sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True)
        self.split=split
        self.count=self.conn.execute('select count(*) from samples where split=?',(split,)).fetchone()[0]
    def __len__(self):return self.count
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self))) ]
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        row=self.conn.execute('select data from samples where split=? and id=?',(self.split,index)).fetchone()
        return pickle.loads(zlib.decompress(row[0]))
    def __iter__(self):
        for i in range(len(self)):yield self[i]

def generate(out,counts):
    out.mkdir(parents=True,exist_ok=True);f=Foundation(ROOT/'1.数据')
    identity={'schema':'tc-mdsm/3','seed':20260920,'counts':counts,'agents':100,'sources':f.source_hashes,'code':code_identity()}
    marker=out/'generation_identity.json'
    def generation_signature(value):
        v=dict(value);v['code']={k:h for k,h in v['code'].items() if k in ('data.py','types.py','world.py')};return v
    if marker.exists() and generation_signature(json.loads(marker.read_text(encoding='utf-8')))!=generation_signature(identity):raise ValueError('dataset generation identity changed; use a new directory')
    atomic_json(marker,identity)
    conn=sqlite3.connect(out/'samples.sqlite');conn.execute('create table if not exists samples(split text,id integer,data blob,primary key(split,id))')
    for split,count in zip(('train','val','test'),counts):
        existing={r[0] for r in conn.execute('select id from samples where split=?',(split,))}
        for i in range(count):
            if i in existing:continue
            e=f.episode(split,i,20260920,agents=100,difficulty=DIFFICULTIES[i%3])[0]
            e.snapshot.states={g:np.asarray(x,dtype=np.float32) for g,x in e.snapshot.states.items()}
            e.snapshot.missing={g:np.asarray(x,dtype=bool) for g,x in e.snapshot.missing.items()}
            conn.execute('insert into samples values(?,?,?)',(split,i,zlib.compress(pickle.dumps(e,protocol=5),1)))
            if (i+1)%100==0:conn.commit();print(json.dumps({'stage':'generate','split':split,'done':i+1,'total':count}),flush=True)
        conn.commit()
    conn.close()
    audit={};templates={};bounds={}
    for split in ('train','val','test'):
        counts_=[];labels=collections.Counter();groups=set();templates[split]=set();times=[]
        for e in Cases(out/'samples.sqlite',split):
            e.snapshot.validate();e.workflow.validate()
            counts_ += [len(e.snapshot.feasible(t)) for t in e.workflow.tasks]
            labels.update((o.kind,str(o.success)) for o in e.observations)
            groups.add(e.provenance['trace_group']);templates[split].add(e.workflow.template)
            times.extend([e.provenance['trace_start'],e.provenance['trace_end']])
        bounds[split]=[min(times),max(times)]
        audit[split]={'workflows':len(Cases(out/'samples.sqlite',split)),'candidate_mean':float(np.mean(counts_)),'candidate_p50':float(np.median(counts_)),'fraction_over32':float(np.mean(np.array(counts_)>32)),'zero_fraction':float(np.mean(np.array(counts_)==0)),'labels':{str(k):v for k,v in labels.items()},'trace_groups':len(groups),'time_bounds':bounds[split]}
    assert not templates['train']&templates['val'] and not templates['train']&templates['test'] and not templates['val']&templates['test']
    assert bounds['train'][1]<bounds['val'][0] and bounds['val'][1]<bounds['test'][0]
    atomic_json(out/'audit.json',audit)
    h=hashlib.sha256()
    with (out/'samples.sqlite').open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    atomic_json(out/'manifest.json',{**identity,'sqlite_sha256':h.hexdigest(),'audit':audit})

def conditions():
    return [('complexity',i,{'difficulty':d,'agents':100}) for i,d in enumerate(DIFFICULTIES)]+[('scale',n,{'difficulty':'中等','agents':n}) for n in [50,100,200,400,800,1600]]+[('concurrency',n,{'difficulty':'中等','agents':100,'batch':n}) for n in [1,2,4,8,16,32,64]]

def evaluate(out,instances,device,seconds):
    freeze=json.loads((out/'冻结协议.json').read_text(encoding='utf-8'))
    if freeze['code']!=code_identity():raise ValueError('source changed since freeze')
    f=Foundation(ROOT/'1.数据')
    if freeze['sources']!=f.source_hashes:raise ValueError('capability sources changed since freeze')
    for seed in freeze['seeds']:
        for method in METHODS:
            slug=method.lower().replace('-','_');checkpoint=out/'训练'/f'seed_{seed}'/slug/'best.pt'
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=freeze['checkpoints'][f'{seed}/{method}']:raise ValueError('checkpoint changed')
            model=load_model(checkpoint,device)
            for block,x,kwargs in conditions():
                for i in range(instances):
                    dest=out/'逐实例结果'/f'seed_{seed}'/slug/f'{block}_{x}_{i}.json'
                    if dest.exists():continue
                    idx=seed_of('formal-v3',block,x,i)%100000000
                    examples=f.episode('test',idx,seed,**kwargs)
                    with torch.no_grad():rows,events,plans=execute_batch(model,examples,time_limit=seconds,seed=seed)
                    for row in rows:row.update(seed=seed,method=method,block=block,x=x,instance=i,trace_group=examples[0].provenance['trace_group'])
                    atomic_json(dest,{'rows':rows,'executions':events,'plans':[asdict(p) for p in plans],'freeze_digest':digest(freeze)})
                    print(json.dumps({'stage':'evaluate','seed':seed,'method':method,'block':block,'x':x,'instance':i,'completed':sum(r['on_time'] for r in rows),'requests':len(rows)}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['generate','train','freeze','evaluate','report','all'])
    parser.add_argument('--output',type=Path,default=ROOT/'6.结果输出'/'2026-09-27_TC-MDSM正式实验'/'运行')
    parser.add_argument('--counts',default='12000,1500,1500');parser.add_argument('--seeds',default='7,17,29,43,61')
    parser.add_argument('--epochs',type=int,default=100);parser.add_argument('--instances',type=int,default=30)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--validation-limit',type=int,default=0);parser.add_argument('--method',choices=METHODS)
    args=parser.parse_args();torch.set_num_threads(2);out=args.output;out.mkdir(parents=True,exist_ok=True)
    counts=list(map(int,args.counts.split(',')));seeds=list(map(int,args.seeds.split(',')))
    if args.stage in ('generate','all'):generate(out/'数据',counts)
    if args.stage in ('train','all'):
        dataset=out/'数据'
        manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'));h=hashlib.sha256()
        with (dataset/'samples.sqlite').open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=manifest['sqlite_sha256']:raise ValueError('dataset checksum mismatch')
        train_cases=Cases(dataset/'samples.sqlite','train');val_cases=Cases(dataset/'samples.sqlite','val')
        for seed in seeds:
            for method in ([args.method] if args.method else METHODS):
                print(json.dumps({'stage':'train_start','seed':seed,'method':method,'device':args.device}),flush=True)
                train(dataset,out/'训练'/f'seed_{seed}'/method.lower().replace('-','_'),method,ModelConfig(),seed=seed,epochs=args.epochs,patience=10,device=args.device,resume=True,validation_limit=args.validation_limit,time_limit=args.seconds,train_cases=train_cases,val_cases=val_cases)
    if args.stage in ('freeze','all'):
        checkpoints={}
        for seed in seeds:
            for method in METHODS:
                ck=out/'训练'/f'seed_{seed}'/method.lower().replace('-','_')/'best.pt'
                if method!='Rule-Latency':
                    last=torch.load(ck.with_name('last.pt'),map_location='cpu',weights_only=False)
                    run=json.loads(ck.with_name('run_manifest.json').read_text(encoding='utf-8'))
                    if run['code']!=code_identity():raise ValueError('training code differs from freezing code')
                    if not (last['epoch']>=args.epochs or (last['epoch']>=7 and last['stale']>=10)):raise ValueError('training has not reached its configured stopping condition')
                    if run['validation_limit']!=args.validation_limit or run['solver_seconds']!=args.seconds:raise ValueError('training validation protocol differs')
                checkpoints[f'{seed}/{method}']=hashlib.sha256(ck.read_bytes()).hexdigest()
        freeze={'seeds':seeds,'instances':args.instances,'epochs':args.epochs,'counts':counts,'hidden':128,'candidate_budget':32,'seconds':args.seconds,'code':code_identity(),'sources':Foundation(ROOT/'1.数据').source_hashes,'checkpoints':checkpoints,'formal':counts==[12000,1500,1500] and seeds==[7,17,29,43,61] and args.instances==30 and args.epochs==50 and args.validation_limit==0 and args.seconds==30}
        dest=out/'冻结协议.json'
        if dest.exists() and json.loads(dest.read_text(encoding='utf-8'))!=freeze:raise ValueError('existing protocol differs')
        atomic_json(dest,freeze)
    if args.stage in ('evaluate','all'):evaluate(out,args.instances,args.device,args.seconds)
    if args.stage in ('report','all'):
        from tc_mdsm_v3_report import report
        report(out)
if __name__=='__main__':main()
