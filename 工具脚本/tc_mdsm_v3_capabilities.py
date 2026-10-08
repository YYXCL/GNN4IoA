"""Versioned reproducible individual capability generation."""
import copy,csv,hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'1.数据'/'3.智能体能力数据'
original=ROOT/'6.结果输出'/'2026-09-20_TC-MDSM旧协议'/'修改前备份'/'3.智能体能力数据'/'agent_capabilities.json'
agents=json.loads(original.read_text(encoding='utf-8-sig'))
rng=random.Random(20260920)
def family(skill):
    name=skill['id'].lower()
    if any(x in name for x in ('video','image','visual','camera','object')):return 'vision'
    if any(x in name for x in ('route','path','plan','dispatch','schedule','navigation')):return 'planning'
    if any(x in name for x in ('detect','anomaly','fault','monitor','sensor','inspect')):return 'detection'
    return 'text'
catalog={}
for a in agents:
    for skill in a['skills']:
        if skill['id'] not in catalog:
            s=copy.deepcopy(skill);s['family']=family(s)
            s['required_tools']=[{'vision':'视频流访问 API','planning':'地图 API','detection':'监测数据库 API','text':'搜索 API'}[s['family']]]
            s['required_knowledge']=['共享业务知识库']
            s['distribution']='common' if int(hashlib.sha256(s['id'].encode()).hexdigest()[:8],16)%10<7 else 'specialist'
            catalog[s['id']]=s
out=[]
for i,a in enumerate(agents):
    b=copy.deepcopy(a);original_ids={s['id'] for s in a['skills']}
    families={f for f in ['vision','planning','detection','text'] if rng.random()<.75}
    ids=original_ids|{sid for sid,s in catalog.items() if s['family'] in families and rng.random()<(.85 if s['distribution']=='common' else .12)}
    b['skills']=[copy.deepcopy(catalog[sid]) for sid in sorted(ids)]
    tools=set(a.get('tool_apis',[]))|{t for s in b['skills'] for t in s['required_tools']}
    b['tool_apis']=sorted(tools);b['工具API权限']='|'.join(sorted(tools));b['接入知识库']='共享业务知识库'
    b['包含的skill']='|'.join(sorted(ids))
    b['能力族']=sorted({s['family'] for s in b['skills']})
    b['物理传感器可用情况']='|'.join(sorted(set(str(a.get('物理传感器可用情况','')).split('|'))|({'摄像头'} if 'vision' in families else set())))
    b['simulation_profile']={'speed':round(rng.uniform(.7,1.35),4),'proficiency':round(rng.uniform(.78,.99),4),'stability':round(rng.uniform(.7,1.3),4),'tool_quota':rng.choice([30,60,120,240])}
    b['工具API速率限制(次/分钟)']=b['simulation_profile']['tool_quota']
    b['能力数据版本']='v3';b['智能体能力描述']+='；共享能力族：'+','.join(b['能力族'])
    out.append(b)
(folder/'agent_capabilities.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
with (folder/'agent_capabilities.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader()
    for a in out:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in a.items()})
manifest={'seed':20260920,'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'agents':len(out),'capability_families':catalog,'assumption':'Synthetic shared capabilities and sensor provisioning, not measurements of real agents.'}
(folder/'能力分布配置.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'agents':len(out),'skills':len(catalog),'common_mean_candidates_first100':sum(sum(sid in {s['id'] for s in a['skills']} for a in out[:100]) for sid,s in catalog.items() if s['distribution']=='common')/sum(s['distribution']=='common' for s in catalog.values())}))
