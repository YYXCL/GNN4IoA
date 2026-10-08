"""First-failure execution: abort pending work and drain running events."""
from __future__ import annotations

import copy
import heapq
from collections import defaultdict
import numpy as np

from .scheduler import plan_with_expansion
from .types import Observation
from .world import World


def execute_batch(model, examples, budgets=(32,), time_limit=30., seed=7, mode="end_to_end"):
    snapshot=copy.deepcopy(examples[0].snapshot)
    workflows=[e.workflow for e in examples]
    worlds={e.workflow.id:World(e.snapshot,e.provenance["world_seed"]) for e in examples}
    wmap={w.id:w for w in workflows}
    fixed=({w.id:{t.id:snapshot.feasible(t) for t in w.tasks} for w in workflows} if mode=="fixed" else None)
    plan,_=plan_with_expansion(model,snapshot,workflows,budgets,time_limit,seed,fixed_candidates=fixed)
    decisions=[plan.elapsed]; plans=[plan]
    now=plan.elapsed
    completed={}; transferred=set(); active={}; heap=[]; pending={}; used=defaultdict(int)
    attempt={w.id:0 for w in workflows}; final_failed={w.id:not plan.admitted[w.id] for w in workflows}
    failure_time={}; recovered=set(); logs=[]; last_finish={}; queue_counter=0
    first_failure={}; failure_reasons={}
    capacities={f"agent:{i}":1 for i in range(len(snapshot.skills))}
    capacities.update({f"compute:{c}:{d}":cap for c,cs in enumerate(snapshot.capacities) for d,cap in enumerate(cs)})
    capacities.update({f"link:{l}":bw for l,bw in enumerate(snapshot.bandwidth)})

    def install(p, selected):
        for key in list(pending):
            if key[1] in selected:
                del pending[key]
        for row in p.assignments:
            wid,k=row["workflow"],row["task"]
            key=("node",wid,k)
            if wid in selected and (wid,k) not in completed and key not in active:
                pending[key]=dict(row)
        for row in p.transfers:
            wid,a,b=row["workflow"],row["parent"],row["child"]
            key=("edge",wid,a,b)
            stamp=(wid,a,b,row["source"],row["target"])
            if wid in selected and stamp not in transferred:
                if key in active:
                    old=active[key]["row"]
                    if (old["source"],old["target"])==(row["source"],row["target"]):
                        continue
                    # Keep the new destination queued while the old transfer drains.
                    # Its eventual outcome is logged but cannot fail the revised DAG.
                    active[key]["superseded"]=True
                pending[key]=dict(row)

    def resources(key,row):
        if key[0]=="edge":
            return [(f"link:{l}",row["demand"]) for l in row["path"]]
        t=wmap[key[1]].tasks[key[2]];i=row["agent"]
        return [(f"agent:{i}",1)]+[(f"compute:{snapshot.agent_compute[i]}:{d}",v) for d,v in enumerate(t.demand)]

    def ready(key,row):
        wid=key[1]
        if key in active or final_failed[wid] or wid in failure_time or row["start"]>now+1e-8:
            return False
        if key[0]=="edge":
            return (wid,row["parent"]) in completed
        for a,b,_ in wmap[wid].edges:
            if b==key[2]:
                if (wid,int(a)) not in completed:
                    return False
                src=completed[wid,int(a)]["agent"]
                if (wid,int(a),int(b),src,row["agent"]) not in transferred:
                    return False
        return True

    def update_feedback(key,row,duration,success):
        # Feedback is published only after completion, never when an outcome is sampled.
        if key[0]=="node":
            i=row["agent"];r=snapshot.states["R"][i][-1]
            count=r[3]+1
            snapshot.states["R"][i]=np.concatenate([np.asarray(snapshot.states["R"][i])[1:],[[
                (r[0]*r[3]+float(success))/count,.9*r[1]+.1*duration,
                .95*r[2]+.05*float(not success),count]]],axis=0).tolist()
        snapshot.revision+=":feedback"

    install(plan,set(wmap))
    while pending or heap or failure_time:
        # Process every completion at this timestamp before starting dependent work.
        while heap and heap[0][0]<=now+1e-8:
            finish,_,key=heapq.heappop(heap)
            record=active.pop(key)
            row,start,duration,success=record["row"],record["start"],record["duration"],record["success"]
            wid=key[1]
            for resource,demand in resources(key,row):
                used[resource]-=demand
            update_feedback(key,row,duration,success)
            if key[0]=="node":
                observation=Observation("node",key[2],row["agent"],duration,success,start,finish,record["attempt"],1.)
                if success:
                    completed[wid,key[2]]={"agent":row["agent"],"end":finish}
                    last_finish[wid]=max(last_finish.get(wid,0),finish)
            else:
                observation=Observation("edge",row["parent"],row["source"],duration,success,start,finish,
                                        record["attempt"],1.,row["child"],row["target"])
                if success:
                    transferred.add((wid,row["parent"],row["child"],row["source"],row["target"]))
            observation.failure_reason=record.get("failure_reason", "")
            logs.append({"workflow":wid,**observation.__dict__})
            if not success and not final_failed[wid] and not record.get("superseded",False):
                final_failed[wid]=True
                first_failure.setdefault(wid,finish)
                failure_reasons.setdefault(wid,record.get("failure_reason", "execution_failure"))
        for key in list(pending):
            if final_failed[key[1]]:
                del pending[key]
        launched=False
        for key,row in sorted(list(pending.items()),key=lambda item:(item[1]["start"],item[0])):
            if not ready(key,row):
                continue
            needs=resources(key,row)
            if any(used[res]+demand>capacities[res] for res,demand in needs):
                continue
            wid=key[1];world=worlds[wid]
            if key[0]=="node":
                duration,success=world.execute(wmap[wid],key[2],row["agent"],now,attempt[wid])
            else:
                mb=next(mb for a,b,mb in wmap[wid].edges if a==row["parent"] and b==row["child"])
                duration,success=world.transfer(wmap[wid],row["parent"],row["child"],row["source"],row["target"],mb,now,attempt[wid])
            for resource,demand in needs:
                used[resource]+=demand
            active[key]={"row":row,"start":now,"duration":duration,"success":success,"attempt":attempt[wid],"failure_reason":world.last_reason}
            queue_counter+=1
            heapq.heappush(heap,(now+duration,queue_counter,key))
            del pending[key];launched=True
        if launched:
            continue
        future=[heap[0][0]] if heap else []
        future += [t for t in failure_time.values() if t>now+1e-8]
        future += [row["start"] for key,row in pending.items() if row["start"]>now+1e-8 and key[1] not in failure_time]
        if not future:
            # Dependency blocked by a terminal failure; preserve in denominator.
            for key in pending:
                final_failed[key[1]]=True
            break
        now=max(now,min(future))
    rows=[]
    for w in workflows:
        complete=all((w.id,t.id) in completed for t in w.tasks) and not final_failed[w.id]
        elapsed=last_finish.get(w.id,0)-w.release if complete else None
        on_time=complete and last_finish[w.id]<=w.deadline
        rows.append({"workflow":w.id,"scenario":w.scenario,"difficulty":w.difficulty,"template":w.template,
                     "on_time":int(on_time),"completed":int(complete),"completion_s":elapsed,
                     "penalized_s":elapsed if on_time else 2*(w.deadline-w.release),
                     "decision_s":decisions[0],"retried":0,
                     "first_failure_s":first_failure.get(w.id),"failure_reason":failure_reasons.get(w.id,plan.reasons.get(w.id,"")),
                     "first_pass_success":int(on_time),
                     "initial_admitted":int(plan.admitted[w.id]),"failure":not complete,
                     "solver_status":plans[-1].status})
    return rows,logs,plans
