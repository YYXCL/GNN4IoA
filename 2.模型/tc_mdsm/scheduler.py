"""Shared CP-SAT backend with optional admission and explicit transfer intervals."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from ortools.sat.python import cp_model

from .types import Plan

class PlanningBudgetExceeded(RuntimeError):
    pass

TICKS = 1000
RISK_SCALE = 1_000_000


def ticks(value):
    return int(math.ceil(value*TICKS-1e-9))


def risk(probability):
    return int(math.ceil(-math.log(max(1e-9,min(1.,probability)))*RISK_SCALE-1e-9))


def solve(snapshot, workflows, predictions, time_limit=30., seed=7, workers=1, completed=None, running=None, warm_start=None):
    started = time.perf_counter()
    completed, running = completed or {}, running or {}
    horizon = max(ticks(w.deadline) for w in workflows)
    model = cp_model.CpModel()
    resources = defaultdict(list)
    admissions, choices, starts, finishes, transfers, total_risks, secondary = {}, {}, {}, {}, {}, {}, []
    capacities = {f"agent:{i}": 1 for i in range(len(snapshot.skills))}
    capacities.update({f"compute:{c}:{d}": cap for c, caps in enumerate(snapshot.capacities) for d,cap in enumerate(caps)})
    capacities.update({f"link:{l}": bw for l,bw in enumerate(snapshot.bandwidth)})
    for reservation in snapshot.reservations:
        lo,hi=ticks(reservation["start"]),ticks(reservation["end"])
        if hi > lo:
            interval=model.new_interval_var(lo,hi-lo,hi,"reserved")
            resources[reservation["resource"]].append((interval,reservation["demand"]))
    for w,p in zip(workflows,predictions):
        if time.perf_counter()-started>=time_limit:raise PlanningBudgetExceeded()
        w.validate(); p.validate()
        if w.id != p.workflow_id:
            raise ValueError("prediction/workflow mismatch")
        accept=model.new_bool_var(f"accept:{w.id}")
        admissions[w.id]=accept
        risks=[]
        for task in w.tasks:
            key=(w.id,task.id)
            done=completed.get(key)
            active=running.get(key)
            candidate_ids=[done["agent"]] if done else [active["agent"]] if active else p.candidates[task.id]
            st=model.new_int_var(0,horizon,f"start:{key}")
            en=model.new_int_var(0,horizon,f"end:{key}")
            starts[key],finishes[key]=st,en
            opts=[]
            for i in candidate_ids:
                if i not in snapshot.feasible(task):
                    raise ValueError("candidate bypassed hard feasibility")
                if not done and (task.id,i) not in p.nodes:
                    continue
                x=model.new_bool_var(f"x:{key}:{i}")
                choices[key,i]=x; opts.append(x)
                duration=0 if done else ticks(p.nodes[task.id,i][0])
                probability=1 if done else p.nodes[task.id,i][1]
                if active:
                    duration=max(1,ticks(active["end"])-ticks(snapshot.time))
                interval=model.new_optional_interval_var(st,duration,en,x,f"run:{key}:{i}")
                if not done:
                    resources[f"agent:{i}"].append((interval,1))
                    for dim,demand in enumerate(task.demand):
                        resources[f"compute:{snapshot.agent_compute[i]}:{dim}"].append((interval,demand))
                    model.add(st>=ticks(max(snapshot.time,w.release,snapshot.available[i]))).only_enforce_if(x)
                if active:
                    model.add(st==ticks(snapshot.time)).only_enforce_if(x)
                risks.append(risk(probability)*x)
            model.add(sum(opts)==accept)
            if done:
                model.add(st==ticks(done["end"])).only_enforce_if(accept)
                model.add(en==ticks(done["end"])).only_enforce_if(accept)
            model.add(st==0).only_enforce_if(accept.Not())
            model.add(en==0).only_enforce_if(accept.Not())
            model.add(en<=int(math.floor(w.deadline*TICKS))).only_enforce_if(accept)
        for parent,child,mb in w.edges:
            parent,child=int(parent),int(child)
            pk,ck=(w.id,parent),(w.id,child)
            if ck in completed or ck in running:
                continue
            parent_ids=[completed[pk]["agent"]] if pk in completed else [running[pk]["agent"]] if pk in running else p.candidates[parent]
            child_ids=p.candidates[child]
            for i in parent_ids:
                if time.perf_counter()-started>=time_limit:raise PlanningBudgetExceeded()
                for j in child_ids:
                    if (pk,i) not in choices or (ck,j) not in choices:
                        continue
                    xi,xj=choices[pk,i],choices[ck,j]
                    edge=(parent,child,i,j)
                    path=snapshot.path(i,j)
                    if path is None or edge not in p.edges:
                        model.add(xi+xj<=1)
                        continue
                    y=model.new_bool_var(f"y:{w.id}:{edge}")
                    model.add(y<=xi);model.add(y<=xj);model.add(y>=xi+xj-1)
                    duration,probability=p.edges[edge]
                    duration=0 if not path else ticks(duration)
                    st=model.new_int_var(0,horizon,"tx_start")
                    en=model.new_int_var(0,horizon,"tx_end")
                    interval=model.new_optional_interval_var(st,duration,en,y,"transfer")
                    model.add(st>=finishes[pk]).only_enforce_if(y)
                    model.add(st>=ticks(snapshot.time)).only_enforce_if(y)
                    model.add(starts[ck]>=en).only_enforce_if(y)
                    demand=min(snapshot.bandwidth[l] for l in path) if path else 0
                    for l in path:
                        resources[f"link:{l}"].append((interval,demand))
                    transfers[w.id,edge]=(y,st,en,path,demand)
                    risks.append(risk(probability)*y)
        risk_bound=(len(w.tasks)+len(w.edges))*risk(1e-9)
        total=model.new_int_var(0,risk_bound,"risk")
        model.add(total==sum(risks))
        total_risks[w.id]=total
        finish=model.new_int_var(0,horizon,"workflow_finish")
        model.add_max_equality(finish,[finishes[w.id,t.id] for t in w.tasks])
        elapsed=model.new_int_var(0,horizon,"elapsed")
        model.add(elapsed==finish-ticks(w.release)*accept)
        norm=model.new_int_var(0,1_000_000,"normalized_makespan")
        # Integer division retains a bounded secondary objective; admission strictly dominates it.
        model.add_division_equality(norm,elapsed*1_000_000,max(1,ticks(w.deadline-w.release)))
        secondary.append(norm+total)
    for name,items in resources.items():
        intervals,demands=zip(*items)
        if name.startswith("agent:"):
            model.add_no_overlap(intervals)
        else:
            model.add_cumulative(intervals,demands,capacities[name])
    upper=sum(1_000_000+(len(w.tasks)+len(w.edges))*risk(1e-9)+1 for w in workflows)
    model.minimize((upper+1)*sum(1-a for a in admissions.values())+sum(secondary))
    invalid=model.validate()
    if invalid:
        raise ValueError(invalid)
    # A fully rejected batch is a valid starting incumbent, avoiding UNKNOWN when
    # the search budget is small. The objective can then improve admission count.
    for index,variable in enumerate(model.proto.variables):
        value=0 if variable.domain[0]<=0<=variable.domain[-1] else variable.domain[0]
        model.add_hint(model.get_int_var_from_proto_index(index),value)
    if warm_start is not None:
        model.clear_hints()
        warm_rows={(r["workflow"],r["task"]):r for r in warm_start.assignments}
        warm_tx={(r["workflow"],(r["parent"],r["child"],r["source"],r["target"])):r for r in warm_start.transfers}
        for wid,var in admissions.items():model.add_hint(var,int(warm_start.admitted[wid]))
        for (key,i),var in choices.items():model.add_hint(var,int(key in warm_rows and warm_rows[key]["agent"]==i))
        for key,var in starts.items():model.add_hint(var,ticks(warm_rows[key]["start"]) if key in warm_rows else 0)
        for key,var in finishes.items():model.add_hint(var,ticks(warm_rows[key]["end"]) if key in warm_rows else 0)
        for key,(selected,st,en,path,demand) in transfers.items():
            r=warm_tx.get(key);model.add_hint(selected,int(r is not None));model.add_hint(st,ticks(r["start"]) if r else 0);model.add_hint(en,ticks(r["end"]) if r else 0)
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=max(.001,time_limit-(time.perf_counter()-started))
    solver.parameters.num_search_workers=workers
    solver.parameters.random_seed=seed
    status=solver.solve(model)
    valid=status in (cp_model.FEASIBLE,cp_model.OPTIMAL)
    admitted={w.id: bool(solver.value(admissions[w.id])) if valid else False for w in workflows}
    rows,tx=[],[]
    if valid:
        for (key,i),x in choices.items():
            if solver.value(x):
                rows.append({"workflow":key[0],"task":key[1],"agent":i,
                             "start":solver.value(starts[key])/TICKS,"end":solver.value(finishes[key])/TICKS,
                             "completed":key in completed,"running":key in running})
        for (wid,edge),(y,st,en,path,demand) in transfers.items():
            if solver.value(y):
                tx.append({"workflow":wid,"parent":edge[0],"child":edge[1],"source":edge[2],"target":edge[3],
                           "start":solver.value(st)/TICKS,"end":solver.value(en)/TICKS,"path":path,"demand":demand})
    plan=Plan(rows,tx,admitted,solver.status_name(status),time.perf_counter()-started,
              {w.id:("" if admitted[w.id] else "no_incumbent" if not valid else "not_admitted_under_constraints") for w in workflows},
              {"objective":solver.objective_value if valid else None,"best_bound":solver.best_objective_bound,
               "workers":workers,"integer_time_unit":"millisecond","objective_rule":"admission then normalized elapsed plus log-risk"})
    for w,p in zip(workflows,predictions):
        if plan.admitted[w.id]:continue
        if any(not snapshot.feasible(t) for t in w.tasks):plan.reasons[w.id]="no_permission_candidate"
        elif any(not any(all(d<=cap for d,cap in zip(t.demand,snapshot.capacities[snapshot.agent_compute[i]])) for i in p.candidates[t.id]) for t in w.tasks):plan.reasons[w.id]="no_resource_compatible_retrieved_candidate"
        elif not valid:plan.reasons[w.id]="solver_no_incumbent"
        else:plan.reasons[w.id]="combined_constraints_or_deadline"
    check_plan(snapshot,workflows,predictions,plan,completed,running)
    return plan


def check_plan(snapshot,workflows,predictions,plan,completed=None,running=None):
    """Independent interval sweep; does not inspect CP-SAT variables."""
    completed,running=completed or {},running or {}
    workflow_map={w.id:w for w in workflows}
    pred_map={p.workflow_id:p for p in predictions}
    rows={(r["workflow"],r["task"]):r for r in plan.assignments}
    if len(rows)!=len(plan.assignments):
        raise AssertionError("duplicate assignment")
    tx={(r["workflow"],r["parent"],r["child"]):r for r in plan.transfers}
    if len(tx)!=len(plan.transfers):
        raise AssertionError("duplicate transfer")
    intervals=defaultdict(list)
    caps={f"agent:{i}":1 for i in range(len(snapshot.skills))}
    caps.update({f"compute:{c}:{d}":cap for c,cs in enumerate(snapshot.capacities) for d,cap in enumerate(cs)})
    caps.update({f"link:{l}":bw for l,bw in enumerate(snapshot.bandwidth)})
    for r in snapshot.reservations:
        intervals[r["resource"]].append((r["start"],r["end"],r["demand"]))
    for wid,admitted in plan.admitted.items():
        w,p=workflow_map[wid],pred_map[wid]
        selected=[r for key,r in rows.items() if key[0]==wid]
        assert len(selected)==(len(w.tasks) if admitted else 0), "partial workflow admission"
        if not admitted:
            continue
        total=0.
        for t in w.tasks:
            key=(wid,t.id);r=rows[key];i=r["agent"]
            assert i in snapshot.feasible(t), "infeasible permission"
            assert r["end"]<=w.deadline+1e-7, "deadline"
            if key in completed:
                assert i==completed[key]["agent"] and abs(r["end"]-completed[key]["end"])<=.0011
                continue
            assert r["start"]+1e-7>=max(w.release,snapshot.time,snapshot.available[i]), "availability"
            duration,prob=p.nodes[t.id,i]
            if key in running:
                assert i==running[key]["agent"] and r["start"]<=snapshot.time+.0011
                duration=max(.001,running[key]["end"]-snapshot.time)
            assert r["end"]-r["start"]+.0011>=duration, "execution duration"
            total-=math.log(prob)
            intervals[f"agent:{i}"].append((r["start"],r["end"],1))
            for d,demand in enumerate(t.demand):
                intervals[f"compute:{snapshot.agent_compute[i]}:{d}"].append((r["start"],r["end"],demand))
        for parent,child,mb in w.edges:
            parent,child=int(parent),int(child)
            if (wid,child) in completed or (wid,child) in running:
                continue
            transfer=tx[wid,parent,child]
            a,b=rows[wid,parent],rows[wid,child]
            assert transfer["source"]==a["agent"] and transfer["target"]==b["agent"]
            path=snapshot.path(a["agent"],b["agent"])
            assert path is not None and path==transfer["path"], "path"
            assert transfer["start"]+1e-7>=max(a["end"],snapshot.time), "transfer precedence"
            assert b["start"]+1e-7>=transfer["end"], "task precedence"
            duration,prob=p.edges[parent,child,a["agent"],b["agent"]]
            if path:
                assert transfer["end"]-transfer["start"]+.0011>=duration
            total-=math.log(prob)
            for l in path:
                intervals[f"link:{l}"].append((transfer["start"],transfer["end"],transfer["demand"]))
        assert math.isfinite(total), "finite soft risk"
    for resource,items in intervals.items():
        events=[]
        for lo,hi,demand in items:
            if hi>lo:
                events.extend([(lo,demand),(hi,-demand)])
        used=0
        for _,delta in sorted(events,key=lambda item:(item[0],item[1])):
            used+=delta
            assert used<=caps[resource], f"capacity violation: {resource}"
    return True


def plan_with_expansion(model,snapshot,workflows,budgets=(32,),time_limit=30.,seed=7,
                        fixed_candidates=None,completed=None,running=None,transferred=None):
    if tuple(budgets)!=(32,):
        raise ValueError("v3 uses a fixed candidate budget of 32")
    started=time.perf_counter();best=None;best_preds=None;attempts=[]
    for budget in budgets:
        remaining=time_limit-(time.perf_counter()-started)
        if remaining<=0:
            break
        predictions=[]
        for w in workflows:
            if time.perf_counter()-started>=time_limit:break
            fixed=fixed_candidates.get(w.id) if fixed_candidates is not None else None
            # Fixed completed/running actions must remain represented, even outside retrieved top-K.
            forced=[(k,i["agent"]) for (wid,k),i in {**(completed or {}),**(running or {})}.items() if wid==w.id]
            if forced and fixed is None:
                preliminary=model.predict(snapshot,w,budget=budget)
                fixed={k:list(v) for k,v in preliminary.candidates.items()}
                for k,i in forced:
                    fixed[k]=[i]+[j for j in fixed[k] if j!=i]
            prediction=model.predict(snapshot,w,budget=budget,fixed_candidates=fixed)
            if transferred:
                prediction.edges=dict(prediction.edges)
                for wid,parent,child,source,target in transferred:
                    key=(parent,child,source,target)
                    if wid==w.id and key in prediction.edges:
                        prediction.edges[key]=(0.,1.)  # Already observed successful delivery.
            predictions.append(prediction)
        remaining=time_limit-(time.perf_counter()-started)
        if remaining<=0 or len(predictions)!=len(workflows):
            break
        exhausted=all(len(snapshot.feasible(t))<=budget for w in workflows for t in w.tasks)
        allocation=max(.001,time_limit-(time.perf_counter()-started))
        try:
            plan=solve(snapshot,workflows,predictions,allocation,seed,completed=completed,running=running)
        except PlanningBudgetExceeded:
            break
        check_plan(snapshot,workflows,predictions,plan)
        attempts.append({"budget":budget,"admitted":sum(plan.admitted.values()),"status":plan.status,
                         "elapsed":plan.elapsed,"prediction_seconds":sum(p.timing.get("total",0) for p in predictions)})
        objective=plan.diagnostics.get("objective")
        score=(sum(plan.admitted.values()),-objective if objective is not None else -float("inf"))
        best_objective=best.diagnostics.get("objective") if best is not None else None
        best_score=(-1,-float("inf")) if best is None else (sum(best.admitted.values()),-best_objective if best_objective is not None else -float("inf"))
        if score>best_score:
            best,best_preds=plan,predictions
        if all(plan.admitted.values()):
            break
        if exhausted:
            break
    if best is None:
        best=Plan([],[],{w.id:False for w in workflows},"BUDGET_EXHAUSTED",time.perf_counter()-started,
                  {w.id:"prediction_budget_exhausted" for w in workflows})
        best_preds=[]
    best.elapsed=time.perf_counter()-started
    best.diagnostics["expansion"]=attempts
    best.diagnostics["predictions"]=[{"workflow":p.workflow_id,"timing":p.timing,"diagnostics":p.diagnostics} for p in best_preds]
    return best,best_preds
