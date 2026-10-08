import copy,itertools,sys
from pathlib import Path
import pytest,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2.模型'))
from tc_mdsm.types import ModelConfig,METHODS,Example,Observation,Snapshot,Task,Workflow,Prediction
from tc_mdsm.model import Predictor,observed_loss
from tc_mdsm.scheduler import solve,check_plan
from tc_mdsm.execution import execute_batch
from tc_mdsm.world import World
torch.set_num_threads(2)

def fixture():
    s=Snapshot(0,[-1.,0.],{
        'A':[[[100,0,30,0,1,0]]*2 for _ in range(2)],
        'C':[[[100,64,32,0,0]]*2],
        'R':[[[.999,2,.001,100]]*2 for _ in range(2)],
        'E':[[[.01,100,.00001,.001]]*2 for _ in range(2)]},
        {g:[[False]*2 for _ in range(n)] for g,n in {'A':2,'C':1,'R':2,'E':2}.items()},
        [['a'],['a']],[['api'],['api']],[[],[]],['x','y'],[0,0],[[2,64,32]],[[0,2],[1,2]],[100,100],[0,0])
    w=Workflow('w',[Task(0,'a','a',['api'],[],100,[1,8,4]),Task(1,'b','a',[],[],100,[1,8,4])],[[0,1,1]],0,20)
    p=Prediction('w',{0:[0,1],1:[0,1]},
                 {(0,0):(2,.999),(0,1):(3,.999),(1,0):(4,.999),(1,1):(1,.999)},
                 {(0,1,i,j):(0 if i==j else .1,.9999) for i in range(2) for j in range(2)})
    return s,w,p

def test_schedule_matches_enumeration():
    s,w,p=fixture();plan=solve(s,[w],[p],2)
    optimum=min(p.nodes[0,i][0]+p.edges[0,1,i,j][0]+p.nodes[1,j][0]
                for i,j in itertools.product(range(2),repeat=2))
    assert plan.admitted['w']
    assert abs(max(r['end'] for r in plan.assignments)-optimum)<.002
    assert check_plan(s,[w],[p],plan)

def test_concurrent_capacity_and_reject():
    s,w,p=fixture();w.tasks=w.tasks[:1];w.edges=[];w.deadline=2.1
    p.candidates={0:[0]};p.nodes={(0,0):(2,.999)};p.edges={}
    w2=copy.deepcopy(w);w2.id='w2';p2=copy.deepcopy(p);p2.workflow_id='w2'
    plan=solve(s,[w,w2],[p,p2],2)
    assert sum(plan.admitted.values())==1
    s.capacities[0][0]=0
    assert not solve(s,[w],[p],2).admitted['w']

def test_checker_detects_tampering():
    s,w,p=fixture();plan=solve(s,[w],[p],2)
    plan.assignments[-1]['start']=0
    with pytest.raises(AssertionError):check_plan(s,[w],[p],plan)

def test_soft_risk_does_not_reject():
    s,w,p=fixture();p.nodes={k:(v[0],.15) for k,v in p.nodes.items()}
    plan=solve(s,[w],[p],2)
    assert plan.admitted[w.id]
    assert check_plan(s,[w],[p],plan)

def test_admission_dominates_soft_risk():
    s,w,p=fixture();p.nodes={k:(v[0],1e-9) for k,v in p.nodes.items()}
    assert solve(s,[w],[p],2).admitted[w.id]

@pytest.mark.parametrize('method',[m for m in METHODS if m!='Rule-Latency'])
def test_gradient_and_inference(method):
    s,w,p=fixture();e=Example(s,w,[Observation('node',0,0,2,True,0,2,0,.5)],{'split':'train'})
    model=Predictor(ModelConfig(hidden=16,top_b=4),method);model.fit_normalizers([e])
    model.train();loss,_=observed_loss(model,e);loss.backward()
    assert torch.isfinite(loss)
    model.eval()
    with torch.no_grad():out=model(s,w)
    assert out['retrieval_scores'] is None
    assert model.sparse==(method=='TC-MDSM')
    assert all(len(v)<=32 for v in out['candidates'].values())

def test_first_failure_terminates_without_replanning(monkeypatch):
    s,w,p=fixture();e=Example(s,w,[],{'world_seed':5})
    class Fake:
        method='test'
        def predict(self,*args,**kwargs):return p
    def fail(self,*args):self.last_reason='injected_failure';return .1,False
    monkeypatch.setattr(World,'execute',fail)
    rows,events,plans=execute_batch(Fake(),[e],time_limit=2)
    assert rows[0]['on_time']==0 and rows[0]['retried']==0 and len(plans)==1
    assert rows[0]['first_failure_s'] is not None
    assert len([x for x in events if x['kind']=='node'])==1

@pytest.mark.parametrize('training',[True,False])
def test_full_state_matches_tc_mdsm_without_gates(training):
    """Only state selection changes; temporal and graph paths remain identical."""
    s,w,_=fixture()
    e=Example(s,w,[Observation('node',0,0,2,True,0,2,0,.5)],{'split':'train'})
    full=Predictor(ModelConfig(hidden=16,top_b=4),'Full-State')
    reference=Predictor(ModelConfig(hidden=16,top_b=4,ablation='no_gates'),'TC-MDSM')
    full.fit_normalizers([e]);reference.load_state_dict(full.state_dict())
    full.train(training);reference.train(training)
    a,b=full(s,w),reference(s,w)
    assert a['candidates']==b['candidates']
    assert a['pairs']==b['pairs'] and a['edge_keys']==b['edge_keys']
    for key in ('node','edge','sparse_loss','gates'):
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    assert torch.all(a['gates']==1) and a['sparse_loss'].item()==0
    if training:
        loss,_=observed_loss(full,e,sparse_weight=.01);loss.backward()
        for prefix in ('grus.A.','grus.C.','layers.0.'):
            assert any(p.grad is not None and p.grad.abs().sum()>0
                       for name,p in full.named_parameters() if name.startswith(prefix))

def test_full_state_uses_historical_observations():
    s,w,_=fixture();e=Example(s,w,[],{'split':'train'})
    model=Predictor(ModelConfig(hidden=16,top_b=4),'Full-State')
    model.fit_normalizers([e]);model.eval()
    changed=copy.deepcopy(s);changed.states['C'][0][0][0]=40
    with torch.no_grad():
        before=model.encode_state(s)[0]['C']
        after=model.encode_state(changed)[0]['C']
    assert not torch.allclose(before,after)

def test_rule_latency_uses_shared_solver():
    s,w,p=fixture();model=Predictor(ModelConfig(hidden=16,top_b=4),'Rule-Latency')
    rows,events,plans=execute_batch(model,[Example(s,w,[],{'world_seed':5})],time_limit=2)
    assert plans[0].status!='GREEDY' and plans[0].admitted[w.id]

def test_future_input_forbidden():
    s,w,p=fixture();s.observed_at[-1]=1
    with pytest.raises(ValueError,match='future'):s.validate()

def test_outcomes_reproducible_and_history_informative():
    s,w,p=fixture();a=World(s,18).execute(w,0,0,0,0);b=World(copy.deepcopy(s),18).execute(w,0,0,0,0)
    assert a==b
    older=copy.deepcopy(s);older.states['C'][0][0][0]=40
    assert World(older,18).execute(w,0,0,0,0)[0]!=a[0]

def test_empty_permissions_and_competing_workflows():
    s,w,p=fixture();s.tools=[[],[]]
    model=Predictor(ModelConfig(hidden=16,top_b=4),'Rule-Latency')
    rows,_,plans=execute_batch(model,[Example(s,w,[],{'world_seed':4})],time_limit=2)
    assert not rows[0]['on_time'] and plans[0].reasons[w.id]=='no_permission_candidate'
    s,w,p=fixture();w2=copy.deepcopy(w);w2.id='w2'
    rows,_,plans=execute_batch(model,[Example(s,w,[],{'world_seed':4}),Example(s,w2,[],{'world_seed':4})],time_limit=2)
    assert len(rows)==2 and rows[0]['decision_s']==rows[1]['decision_s']

def test_checkpoint_resume_reproduces_training(tmp_path):
    from tc_mdsm.training import train
    (tmp_path/'manifest.json').write_text('{}',encoding='utf-8')
    s,w,p=fixture();e=Example(s,w,[Observation('node',0,0,2,True,0,2,0,.5)],{'split':'train','world_seed':8,'trace_group':'train:1'})
    v=copy.deepcopy(e);v.provenance.update(split='val',trace_group='val:1')
    kwargs=dict(method='TC-MDSM',config=ModelConfig(hidden=16,top_b=4),seed=7,patience=10,time_limit=1,train_cases=[e],val_cases=[v])
    train(tmp_path,tmp_path/'full',epochs=2,**kwargs)
    train(tmp_path,tmp_path/'resume',epochs=1,**kwargs)
    train(tmp_path,tmp_path/'resume',epochs=2,resume=True,**kwargs)
    a=torch.load(tmp_path/'full'/'last.pt',weights_only=False);b=torch.load(tmp_path/'resume'/'last.pt',weights_only=False)
    for key,value in a['model'].items():assert torch.equal(value,b['model'][key]),key
