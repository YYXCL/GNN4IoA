"""资源竞争下的离散事件执行：首次失败终止请求，不进行重规划。"""
import heapq
from collections import defaultdict

from .scheduler import plan_workflows
from .world import World


def execute_batch(model, examples, budgets=None, time_limit=30., seed=7):
    snapshot=examples[0].snapshot
    workflows={e.workflow.id:e.workflow for e in examples}
    worlds={e.workflow.id:World(e.snapshot,e.provenance["world_seed"]) for e in examples}
    plan=plan_workflows(model,snapshot,list(workflows.values()),budgets,time_limit,seed)
    now=plan.elapsed
    pending={}
    for row in plan.assignments:
        pending["node",row["workflow"],row["task"]]=row
    for row in plan.transfers:
        pending["edge",row["workflow"],row["parent"],row["child"]]=row
    finished={};transferred=set();active={};heap=[];events=[];used=defaultdict(int)
    failed={wid:not admitted for wid,admitted in plan.admitted.items()}
    reasons=dict(plan.reasons);first_failure={};last_finish={};counter=0
    capacities={f"agent:{i}":1 for i in range(len(snapshot.skills))}
    capacities.update({f"compute:{c}:{d}":cap for c,cs in enumerate(snapshot.capacities) for d,cap in enumerate(cs)})
    capacities.update({f"link:{l}":bw for l,bw in enumerate(snapshot.bandwidth)})

    def resources(key,row):
        if key[0]=="edge":
            return [(f"link:{l}",row["demand"]) for l in row["path"]]
        task=workflows[key[1]].tasks[key[2]]
        return [(f"agent:{row['agent']}",1)]+[(f"compute:{snapshot.agent_compute[row['agent']]}:{d}",v) for d,v in enumerate(task.demand)]

    def ready(key,row):
        wid=key[1]
        if failed[wid] or row["start"]>now+1e-8:
            return False
        if key[0]=="edge":
            return (wid,row["parent"]) in finished
        for parent,child,_ in workflows[wid].edges:
            if child==key[2]:
                parent=int(parent);child=int(child)
                if (wid,parent) not in finished:
                    return False
                if (wid,parent,child,finished[wid,parent],row["agent"]) not in transferred:
                    return False
        return True

    while pending or heap:
        while heap and heap[0][0]<=now+1e-8:
            end,_,key=heapq.heappop(heap)
            row,start,duration,success,reason=active.pop(key)
            wid=key[1]
            for resource,demand in resources(key,row):used[resource]-=demand
            event={"workflow":wid,"kind":key[0],"started":start,"finished":end,
                   "duration":duration,"success":success,"failure_reason":reason}
            if key[0]=="node":
                event.update(task=key[2],agent=row["agent"])
                if success:
                    finished[wid,key[2]]=row["agent"]
                    last_finish[wid]=max(last_finish.get(wid,0),end)
            else:
                event.update(task=row["parent"],target_task=row["child"],agent=row["source"],target_agent=row["target"])
                if success:transferred.add((wid,row["parent"],row["child"],row["source"],row["target"]))
            events.append(event)
            if not success and not failed[wid]:
                failed[wid]=True;first_failure[wid]=end;reasons[wid]=reason
        pending={key:row for key,row in pending.items() if not failed[key[1]]}
        launched=False
        for key,row in sorted(list(pending.items()),key=lambda item:(item[1]["start"],item[0])):
            needs=resources(key,row)
            if not ready(key,row) or any(used[r]+d>capacities[r] for r,d in needs):continue
            wid=key[1];world=worlds[wid];workflow=workflows[wid]
            if key[0]=="node":
                duration,success=world.execute(workflow,key[2],row["agent"],now,0)
            else:
                mb=next(mb for a,b,mb in workflow.edges if a==row["parent"] and b==row["child"])
                duration,success=world.transfer(workflow,row["parent"],row["child"],row["source"],row["target"],mb,now,0)
            for resource,demand in needs:used[resource]+=demand
            active[key]=(row,now,duration,success,world.last_reason)
            counter+=1;heapq.heappush(heap,(now+duration,counter,key))
            del pending[key];launched=True
        if launched:continue
        future=[heap[0][0]] if heap else []
        future.extend(row["start"] for row in pending.values() if row["start"]>now+1e-8)
        if not future:
            for key in pending:failed[key[1]]=True;reasons[key[1]]="blocked_dependency"
            break
        now=max(now,min(future))
    rows=[]
    for wid,w in workflows.items():
        complete=all((wid,t.id) in finished for t in w.tasks) and not failed[wid]
        elapsed=last_finish[wid]-w.release if complete else None
        on_time=complete and last_finish[wid]<=w.deadline
        reason=reasons[wid] if not complete else "" if on_time else "deadline_missed"
        rows.append({"workflow":wid,"scenario":w.scenario,"difficulty":w.difficulty,"template":w.template,
                     "on_time":int(on_time),"completed":int(complete),"completion_s":elapsed,
                     "penalized_s":elapsed if on_time else 2*(w.deadline-w.release),
                     "decision_s":plan.elapsed,"initial_admitted":int(plan.admitted[wid]),
                     "first_failure_s":first_failure.get(wid),"failure_reason":reason,"solver_status":plan.status})
    return rows,events,[plan]
