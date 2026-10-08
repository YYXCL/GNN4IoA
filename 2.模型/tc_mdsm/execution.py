"""第4步：首次失败即终止的执行（！）模型：一旦失败就中止待执行工作，并排空正在运行的事件。"""
from __future__ import annotations

import copy
import heapq
from collections import defaultdict
import numpy as np

from .scheduler import plan_with_expansion
from .types import Observation
from .world import World


def execute_batch(model, examples, budgets=(32,), time_limit=30., seed=7, mode="end_to_end"):
    """对一个批次的工作流做调度 + 离散事件仿真执行，返回结果行、事件日志与方案列表。"""
    snapshot=copy.deepcopy(examples[0].snapshot)     # 复制快照（执行过程中会更新反馈）
    workflows=[e.workflow for e in examples]
    worlds={e.workflow.id:World(e.snapshot,e.provenance["world_seed"]) for e in examples}
    wmap={w.id:w for w in workflows}
    # fixed 模式：候选固定为全部可行智能体，用于消融实验
    fixed=({w.id:{t.id:snapshot.feasible(t) for t in w.tasks} for w in workflows} if mode=="fixed" else None)
    plan,_=plan_with_expansion(model,snapshot,workflows,budgets,time_limit,seed,fixed_candidates=fixed)
    decisions=[plan.elapsed]; plans=[plan]
    now=plan.elapsed
    # 离散事件仿真的状态：已完成、已传输、进行中、待执行事件堆、资源占用等
    completed={}; transferred=set(); active={}; heap=[]; pending={}; used=defaultdict(int)
    attempt={w.id:0 for w in workflows}; final_failed={w.id:not plan.admitted[w.id] for w in workflows}
    failure_time={}; recovered=set(); logs=[]; last_finish={}; queue_counter=0
    first_failure={}; failure_reasons={}
    # 各类资源容量
    capacities={f"agent:{i}":1 for i in range(len(snapshot.skills))}
    capacities.update({f"compute:{c}:{d}":cap for c,cs in enumerate(snapshot.capacities) for d,cap in enumerate(cs)})
    capacities.update({f"link:{l}":bw for l,bw in enumerate(snapshot.bandwidth)})

    def install(p, selected):
        """把方案 p 中属于 selected 工作流的分配/传输装入待执行队列。"""
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
                    # 旧传输排空期间，新的目的地先排队；其结果会被记录但不会让修订后的 DAG 失败。
                    active[key]["superseded"]=True
                pending[key]=dict(row)

    def resources(key,row):
        """返回某动作需要占用的资源及需求量。"""
        if key[0]=="edge":
            return [(f"link:{l}",row["demand"]) for l in row["path"]]
        t=wmap[key[1]].tasks[key[2]];i=row["agent"]
        return [(f"agent:{i}",1)]+[(f"compute:{snapshot.agent_compute[i]}:{d}",v) for d,v in enumerate(t.demand)]

    def ready(key,row):
        """判断某动作现在是否满足启动条件（未被阻塞、依赖已完成、时间已到）。"""
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
        """动作完成后把真实观测回写到 R 组状态，供后续决策使用。"""
        if key[0]=="node":
            i=row["agent"];r=snapshot.states["R"][i][-1]
            count=r[3]+1
            snapshot.states["R"][i]=np.concatenate([np.asarray(snapshot.states["R"][i])[1:],[[
                (r[0]*r[3]+float(success))/count,.9*r[1]+.1*duration,
                .95*r[2]+.05*float(not success),count]]],axis=0).tolist()
        snapshot.revision+=":feedback"    # 修订号变化 -> 缓存失效

    install(plan,set(wmap))
    while pending or heap or failure_time:
        # 先把当前时间戳上的所有完成事件处理完，再启动依赖它们的动作
        while heap and heap[0][0]<=now+1e-8:
            finish,_,key=heapq.heappop(heap)
            record=active.pop(key)
            row,start,duration,success=record["row"],record["start"],record["duration"],record["success"]
            wid=key[1]
            for resource,demand in resources(key,row):
                used[resource]-=demand     # 释放资源
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
                # 首次失败即判定该工作流失败（记录失败时间与原因）
                final_failed[wid]=True
                first_failure.setdefault(wid,finish)
                failure_reasons.setdefault(wid,record.get("failure_reason", "execution_failure"))
        for key in list(pending):
            if final_failed[key[1]]:
                del pending[key]
        launched=False
        # 按计划开始时间顺序尝试启动可运行的动作
        for key,row in sorted(list(pending.items()),key=lambda item:(item[1]["start"],item[0])):
            if not ready(key,row):
                continue
            needs=resources(key,row)
            if any(used[res]+demand>capacities[res] for res,demand in needs):
                continue     # 资源不足，稍后再试
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
            heapq.heappush(heap,(now+duration,queue_counter,key))   # 结束后触发完成事件
            del pending[key];launched=True
        if launched:
            continue
        # 没有可启动动作时，时间推进到下一个"未来事件"
        future=[heap[0][0]] if heap else []
        future += [t for t in failure_time.values() if t>now+1e-8]
        future += [row["start"] for key,row in pending.items() if row["start"]>now+1e-8 and key[1] not in failure_time]
        if not future:
            # 因终态失败导致依赖阻塞：保留在分母中（记为失败）
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
