from pathlib import Path
import collections,copy,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'2.模型'))
import numpy as np
from tc_mdsm.data import Foundation
from tc_mdsm.world import World
from tc_mdsm.types import ModelConfig
from tc_mdsm.model import Predictor
from tc_mdsm.execution import execute_batch
from tc_mdsm.training import atomic_json
f=Foundation(ROOT/'1.数据');m=Predictor(ModelConfig(),'Rule-Latency')
rows=[];durations=[];history_changes=[];noise=[];reasons=collections.Counter()
for i in range(30):
    e=f.episode('val',50000+i,20260920,agents=100,difficulty='中等')[0]
    r,events,_=execute_batch(m,[e],time_limit=30);rows+=r
    reasons.update(x['failure_reason'] for x in events if not x['success'])
    task=e.workflow.tasks[0];eligible=e.snapshot.feasible(task)
    samples=[World(e.snapshot,e.provenance['world_seed']).execute(e.workflow,0,a,0,0)[0] for a in eligible]
    if samples:durations.append(float(np.percentile(samples,90)/max(.001,np.percentile(samples,10))))
    if eligible:
        a=eligible[0];s=copy.deepcopy(e.snapshot);c=s.agent_compute[a]
        s.states['C'][c][:-1]=list(reversed(s.states['C'][c][:-1]))
        original=World(e.snapshot,e.provenance['world_seed']).execute(e.workflow,0,a,0,0)[0]
        changed=World(s,e.provenance['world_seed']).execute(e.workflow,0,a,0,0)[0]
        history_changes.append(abs(original-changed))
        repeats=[World(e.snapshot,100+j).execute(e.workflow,0,a,0,0)[0] for j in range(8)]
        noise.append(float(np.std(repeats)))
result={'purpose':'validation-only environment audit; not a method ranking experiment','requests':len(rows),'admitted':sum(r['initial_admitted'] for r in rows),'first_pass_on_time':sum(r['on_time'] for r in rows),'execution_failed_after_admission':sum(r['first_failure_s'] is not None for r in rows),'failure_reasons':dict(reasons),'candidate_duration_p90_p10_mean':float(np.mean(durations)),'same_current_reordered_history_absolute_duration_change_mean_s':float(np.mean(history_changes)),'exogenous_seed_duration_std_mean_s':float(np.mean(noise)),'limitations':'Synthetic half-second service integration and reserved link capacity. Noise statistic includes external dynamics and execution noise, not a causal variance decomposition.'}
dest=ROOT/'6.结果输出'/'2026-09-27_TC-MDSM正式实验'/'环境审计.json';atomic_json(dest,result);print(json.dumps(result),flush=True)
