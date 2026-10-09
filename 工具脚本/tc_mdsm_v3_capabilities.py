""""智能体个体能力"数据生成脚本。
作用：基于旧协议下的能力数据，合成一套 v3 能力数据（含共享能力族、工具/知识库需求、仿真行为参数与速率限制），并输出 JSON、CSV 与配置清单。
注意：其中的共享能力与传感器配置均为假设，并非对真实智能体的测量。
"""
import copy,csv,hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]                       # 仓库根目录（本文件所在目录的上一级）
folder=ROOT/'1.数据'/'3.智能体能力数据'                           # 本脚本的输出目录
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True,help='待扩展的基础能力 JSON')
original=parser.parse_args().source
agents=json.loads(original.read_text(encoding='utf-8-sig'))    # 读取原始智能体列表（utf-8-sig 以兼容 BOM）
rng=random.Random(20260920)                                    # 固定随机种子，保证结果可复现

def family(skill):
    """根据技能 id 中的关键词，把技能归入四个能力族之一：vision/planning/detection/text。"""
    name=skill['id'].lower()
    if any(x in name for x in ('video','image','visual','camera','object')):return 'vision'
    if any(x in name for x in ('route','path','plan','dispatch','schedule','navigation')):return 'planning'
    if any(x in name for x in ('detect','anomaly','fault','monitor','sensor','inspect')):return 'detection'
    return 'text'

# 汇总全量技能目录：为每个技能补上能力族、所需工具/知识库，以及"通用/专家"分布标记
catalog={}
for a in agents:
    for skill in a['skills']:
        if skill['id'] not in catalog:
            s=copy.deepcopy(skill);s['family']=family(s)
            # 每个能力族对应一个所需的工具 API
            s['required_tools']=[{'vision':'视频流访问 API','planning':'地图 API','detection':'监测数据库 API','text':'搜索 API'}[s['family']]]
            s['required_knowledge']=['共享业务知识库']
            # 用技能 id 的哈希决定分布：约 70% 为 common（通用），其余为 specialist（专家）
            s['distribution']='common' if int(hashlib.sha256(s['id'].encode()).hexdigest()[:8],16)%10<7 else 'specialist'
            catalog[s['id']]=s

out=[]
for i,a in enumerate(agents):
    b=copy.deepcopy(a);original_ids={s['id'] for s in a['skills']}       # 该智能体原有的技能集合
    # 随机挑选纳入的能力族（每个族约 75% 概率被选中）
    families={f for f in ['vision','planning','detection','text'] if rng.random()<.75}
    # 合并原有技能 + 从目录中按分布概率新增的技能（common 85%，specialist 12%）
    ids=original_ids|{sid for sid,s in catalog.items() if s['family'] in families and rng.random()<(.85 if s['distribution']=='common' else .12)}
    b['skills']=[copy.deepcopy(catalog[sid]) for sid in sorted(ids)]
    # 工具 API 权限 = 原有工具 ∪ 新技能所需工具
    tools=set(a.get('tool_apis',[]))|{t for s in b['skills'] for t in s['required_tools']}
    b['tool_apis']=sorted(tools);b['工具API权限']='|'.join(sorted(tools));b['接入知识库']='共享业务知识库'
    b['包含的skill']='|'.join(sorted(ids))
    b['能力族']=sorted({s['family'] for s in b['skills']})
    b['物理传感器可用情况']='|'.join(sorted(set(str(a.get('物理传感器可用情况','')).split('|'))|({'摄像头'} if 'vision' in families else set())))
    # 仿真行为参数（速度/熟练度/稳定性）与工具调用配额，均为合成值
    b['simulation_profile']={'speed':round(rng.uniform(.7,1.35),4),'proficiency':round(rng.uniform(.78,.99),4),'stability':round(rng.uniform(.7,1.3),4),'tool_quota':rng.choice([30,60,120,240])}
    b['工具API速率限制(次/分钟)']=b['simulation_profile']['tool_quota']
    b['能力数据版本']='v3';b['智能体能力描述']+='；共享能力族：'+','.join(b['能力族'])
    out.append(b)

# 写出 JSON 主数据
(folder/'agent_capabilities.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
# 写出 CSV（把 dict/list 字段序列化为 JSON 字符串），utf-8-sig 便于 Excel 直接打开
with (folder/'agent_capabilities.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader()
    for a in out:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in a.items()})
print(f'生成 {len(out)} 个智能体，输出到 {folder}')
