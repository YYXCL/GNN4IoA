"""第3步：共享的 CP-SAT 调度后端：支持可选的任务接纳与显式传输时间区间。"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from ortools.sat.python import cp_model

from .types import Plan

class PlanningBudgetExceeded(RuntimeError):
    pass

TICKS = 1000            # 时间刻度：1 秒 = 1000 tick（毫秒）
RISK_SCALE = 1_000_000  # 风险（负对数概率）的整数放大系数


def ticks(value):
    """把秒向上取整转换为 tick。"""
    return int(math.ceil(value*TICKS-1e-9))


def risk(probability):
    """把成功概率转换为整数风险值（-log p），便于在 CP-SAT 中用整数线性表达。"""
    return int(math.ceil(-math.log(max(1e-9,min(1.,probability)))*RISK_SCALE-1e-9))


def solve(snapshot, workflows, predictions, time_limit=30., seed=7, workers=1, reserved=None, timings=None):
    """用 CP-SAT 求解调度：先最大化接纳数，再最小化归一化完成时间与风险。"""
    started = time.perf_counter()
    timings = timings if timings is not None else {}
    horizon = max([ticks(w.deadline) for w in workflows]+[ticks(r['end']) for r in (reserved.assignments if reserved else [])])
    model = cp_model.CpModel()
    resources = defaultdict(list)   # 资源名 -> [(区间变量, 需求)]
    admissions, choices, starts, finishes, transfers, total_risks, secondary = {}, {}, {}, {}, {}, {}, []
    # 各类资源的容量：智能体=1，算力节点按 [GPU槽位,内存,显存]，链路按带宽
    capacities = {f"agent:{i}": 1 for i in range(len(snapshot.skills))}
    capacities.update({f"compute:{c}:{d}": cap for c, caps in enumerate(snapshot.capacities) for d,cap in enumerate(caps)})
    capacities.update({f"link:{l}": bw for l,bw in enumerate(snapshot.bandwidth)})
    # 先前分批得到的方案作为固定区间，后续批次不能重复占用资源。
    if reserved is not None:
        for row in reserved.assignments:
            st,en=ticks(row['start']),ticks(row['end'])
            interval=model.new_interval_var(st,en-st,en,'reserved_node')
            resources[f"agent:{row['agent']}"].append((interval,1))
            for dim,demand in enumerate(row['demand']):
                resources[f"compute:{snapshot.agent_compute[row['agent']]}:{dim}"].append((interval,demand))
        for row in reserved.transfers:
            if not row['path']:continue
            st,en=ticks(row['start']),ticks(row['end'])
            interval=model.new_interval_var(st,en-st,en,'reserved_transfer')
            for link in row['path']:resources[f'link:{link}'].append((interval,row['demand']))
    for w,p in zip(workflows,predictions):
        if time.perf_counter()-started>=time_limit:raise PlanningBudgetExceeded()
        accept=model.new_bool_var(f"accept:{w.id}")   # 是否接纳该工作流
        admissions[w.id]=accept
        risks=[]
        for task in w.tasks:
            key=(w.id,task.id)
            candidate_ids=p.candidates[task.id]
            st=model.new_int_var(0,horizon,f"start:{key}")
            en=model.new_int_var(0,horizon,f"end:{key}")
            starts[key],finishes[key]=st,en
            opts=[]
            for i in candidate_ids:
                if (task.id,i) not in p.nodes:
                    continue
                x=model.new_bool_var(f"x:{key}:{i}")   # 是否把任务分配给智能体 i
                choices[key,i]=x; opts.append(x)
                duration=ticks(p.nodes[task.id,i][0])
                probability=p.nodes[task.id,i][1]
                interval=model.new_optional_interval_var(st,duration,en,x,f"run:{key}:{i}")
                # 占用智能体槽位与算力节点资源
                resources[f"agent:{i}"].append((interval,1))
                for dim,demand in enumerate(task.demand):
                    resources[f"compute:{snapshot.agent_compute[i]}:{dim}"].append((interval,demand))
                model.add(st>=ticks(max(snapshot.time,w.release,snapshot.available[i]))).only_enforce_if(x)
                risks.append(risk(probability)*x)
            model.add(sum(opts)==accept)   # 被接纳时恰好选一个候选
            model.add(st==0).only_enforce_if(accept.Not())
            model.add(en==0).only_enforce_if(accept.Not())
            model.add(en<=int(math.floor(w.deadline*TICKS))).only_enforce_if(accept)   # 截止时间约束
        for parent,child,mb in w.edges:
            # 依赖边：若父子分别选了 i、j，则需要一次传输区间，且子任务在传输结束后才能开始
            parent,child=int(parent),int(child)
            pk,ck=(w.id,parent),(w.id,child)
            parent_ids=p.candidates[parent]
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
                        model.add(xi+xj<=1)   # 不可达或未预测：禁止该组合
                        continue
                    y=model.new_bool_var(f"y:{w.id}:{edge}")   # y = xi AND xj
                    model.add(y<=xi);model.add(y<=xj);model.add(y>=xi+xj-1)
                    if not path:
                        model.add(starts[ck]>=finishes[pk]).only_enforce_if(y)
                        transfers[w.id,edge]=(y,finishes[pk],finishes[pk],[],0)
                        continue
                    duration,probability=p.edges[edge]
                    duration=0 if not path else ticks(duration)
                    st=model.new_int_var(0,horizon,"tx_start")
                    en=model.new_int_var(0,horizon,"tx_end")
                    interval=model.new_optional_interval_var(st,duration,en,y,"transfer")
                    model.add(st>=finishes[pk]).only_enforce_if(y)   # 传输在父任务完成后开始
                    model.add(st>=ticks(snapshot.time)).only_enforce_if(y)
                    model.add(starts[ck]>=en).only_enforce_if(y)     # 子任务在传输完成后开始
                    demand=min(snapshot.bandwidth[l] for l in path) if path else 0
                    for l in path:
                        resources[f"link:{l}"].append((interval,demand))
                    transfers[w.id,edge]=(y,st,en,path,demand)
                    risks.append(risk(probability)*y)
        # 该工作流的总风险（负对数成功率之和）
        risk_bound=(len(w.tasks)+len(w.edges))*risk(1e-9)
        total=model.new_int_var(0,risk_bound,"risk")
        model.add(total==sum(risks))
        total_risks[w.id]=total
        finish=model.new_int_var(0,horizon,"workflow_finish")
        model.add_max_equality(finish,[finishes[w.id,t.id] for t in w.tasks])
        elapsed=model.new_int_var(0,horizon,"elapsed")
        model.add(elapsed==finish-ticks(w.release)*accept)
        norm=model.new_int_var(0,1_000_000,"normalized_makespan")
        # 整数除法得到有界的次级目标；接纳数严格优先于它。
        model.add_division_equality(norm,elapsed*1_000_000,max(1,ticks(w.deadline-w.release)))
        secondary.append(norm+total)
    # 资源容量约束：智能体互斥（no_overlap），其余资源用累积约束（cumulative）
    for name,items in resources.items():
        intervals,demands=zip(*items)
        if name.startswith("agent:"):
            model.add_no_overlap(intervals)
        else:
            model.add_cumulative(intervals,demands,capacities[name])
    # 目标：先最大化接纳数（权重远大于次级项），再最小化完成时间与风险
    upper=sum(1_000_000+(len(w.tasks)+len(w.edges))*risk(1e-9)+1 for w in workflows)
    model.minimize((upper+1)*sum(1-a for a in admissions.values())+sum(secondary))
    # 全拒绝是一个合法初始解，避免搜索预算很小时返回 UNKNOWN；目标随后可提升接纳数。
    for index,variable in enumerate(model.proto.variables):
        value=0 if variable.domain[0]<=0<=variable.domain[-1] else variable.domain[0]
        model.add_hint(model.get_int_var_from_proto_index(index),value)
    solver=cp_model.CpSolver()
    build_elapsed=time.perf_counter()-started
    timings['constraint_build_s']=timings.get('constraint_build_s',0.)+build_elapsed
    solver.parameters.max_time_in_seconds=max(.001,time_limit-(time.perf_counter()-started))
    solver.parameters.num_search_workers=workers
    solver.parameters.random_seed=seed
    solving=time.perf_counter()
    status=solver.solve(model)
    timings['solver_s']=timings.get('solver_s',0.)+time.perf_counter()-solving
    extracting=time.perf_counter()
    valid=status in (cp_model.FEASIBLE,cp_model.OPTIMAL)
    admitted={w.id: bool(solver.value(admissions[w.id])) if valid else False for w in workflows}
    rows,tx=[],[]
    if valid:
        tasks={(w.id,t.id):t for w in workflows for t in w.tasks}
        # 抽取被选中的分配与传输
        for (key,i),x in choices.items():
            if solver.value(x):
                rows.append({"workflow":key[0],"task":key[1],"agent":i,
                             "start":solver.value(starts[key])/TICKS,"end":solver.value(finishes[key])/TICKS,
                             "demand":tasks[key].demand})
        for (wid,edge),(y,st,en,path,demand) in transfers.items():
            if solver.value(y):
                tx.append({"workflow":wid,"parent":edge[0],"child":edge[1],"source":edge[2],"target":edge[3],
                           "start":solver.value(st)/TICKS,"end":solver.value(en)/TICKS,"path":path,"demand":demand})
    reasons={w.id:"" if admitted[w.id] else "solver_no_incumbent" if not valid
             else "combined_constraints_or_deadline" for w in workflows}
    for w,p in zip(workflows,predictions):
        if not admitted[w.id] and any(not p.candidates[t.id] for t in w.tasks):
            reasons[w.id]="no_permission_candidate"
    timings['solution_extract_s']=timings.get('solution_extract_s',0.)+time.perf_counter()-extracting
    return Plan(rows,tx,admitted,solver.status_name(status),time.perf_counter()-started,reasons)


def plan_workflows(model,snapshot,workflows,budgets=None,time_limit=30.,seed=7):
    """提前准备全局表示，在共享在线预算下分批预测/规划，保留已得到的方案。"""
    preparing=time.perf_counter()
    model.prepare_snapshot(snapshot)
    timings={'global_prepare_s':time.perf_counter()-preparing,'online_prediction_s':0.,
             'constraint_build_s':0.,'solver_s':0.,'solution_extract_s':0.}
    started=time.perf_counter()
    budget=(budgets or model.config.candidate_budgets)[0]
    size=model.config.planning_batch_size
    chunks=[workflows[i:i+size] for i in range(0,len(workflows),size)]
    combined=Plan([],[],{w.id:False for w in workflows},'BUDGET_EXHAUSTED',0.,
                  {w.id:'planning_budget_exhausted' for w in workflows},timings)
    statuses=[]
    for index,chunk in enumerate(chunks):
        remaining=time_limit-(time.perf_counter()-started)
        if remaining<=0:break
        # 各批次分享剩余时间，防止第一批独占全部搜索预算。
        allotted=remaining/(len(chunks)-index)
        chunk_started=time.perf_counter();predictions=[]
        for w in chunk:
            if time.perf_counter()-chunk_started>=allotted:break
            predictions.append(model.predict(snapshot,w,budget=budget))
        prediction_elapsed=time.perf_counter()-chunk_started
        timings['online_prediction_s']+=prediction_elapsed
        if len(predictions)!=len(chunk) or prediction_elapsed>=allotted:
            statuses.append('BUDGET_EXHAUSTED');continue
        building=time.perf_counter()
        try:
            plan=solve(snapshot,chunk,predictions,allotted-prediction_elapsed,seed,
                       reserved=combined,timings=timings)
        except PlanningBudgetExceeded:
            timings['constraint_build_s']+=time.perf_counter()-building
            statuses.append('BUDGET_EXHAUSTED');continue
        statuses.append(plan.status)
        combined.assignments.extend(plan.assignments);combined.transfers.extend(plan.transfers)
        combined.admitted.update(plan.admitted);combined.reasons.update(plan.reasons)
    combined.status=(statuses[0] if len(chunks)==1 and statuses else
                     'BATCH_FEASIBLE' if all(combined.admitted.values()) else
                     'PARTIAL' if any(combined.admitted.values()) else
                     'BUDGET_EXHAUSTED' if 'BUDGET_EXHAUSTED' in statuses or len(statuses)<len(chunks) else
                     'BATCH_NO_ADMISSION')
    combined.elapsed=time.perf_counter()-started
    return combined
