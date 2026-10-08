"""Independent dynamic environment; no policy identity or predicted labels are read.

Integration uses half-second steps. The last partial step is prorated. Background
service and tool-health trajectories are common random processes across methods.
Reservations in execution.py handle competition between admitted tasks.
"""
from __future__ import annotations
import math
import numpy as np
from .data import seed_of

class World:
    def __init__(self,snapshot,seed):
        self.snapshot,self.seed=snapshot,seed
        self.last_reason=''
    def rng(self,*key):
        return np.random.default_rng(seed_of(self.seed,*key))
    def profile(self,i):
        return self.snapshot.agent_profiles[i] if self.snapshot.agent_profiles else {}
    def wave(self,key,t):
        phase=(seed_of(self.seed,'phase',key)%100000)/100000*2*math.pi
        return math.sin(t/12+phase)+.4*math.sin(t/3.7+phase*2)
    def integrate(self,amount,start,rate,limit=1200.):
        elapsed=0.
        while amount>1e-9 and elapsed<limit:
            speed=max(.01,float(rate(start+elapsed)))
            step=min(.5,amount/speed)
            amount-=speed*step;elapsed+=step
        return max(.001,elapsed),amount<=1e-9
    def execute(self,workflow,task,agent,started,attempt):
        s=self.snapshot;t=workflow.tasks[task];c=s.agent_compute[agent]
        hist=np.asarray(s.states['C'][c]);cur=hist[-1]
        trend=(cur[0]-hist[max(0,len(hist)-4),0])/max(1,min(3,len(hist)-1))
        p=self.profile(agent);speed=p.get('speed',1.)
        rng=self.rng('node',workflow.id,task,agent,attempt)
        noise=float(rng.lognormal(-.005,.10))
        def rate(now):
            base=max(4.,cur[0]+trend*min(20.,max(0.,now)))
            return base*speed*max(.15,1+.30*self.wave(('compute',c),now))/noise
        duration,finished=self.integrate(t.work,started,rate)
        rh=np.asarray(s.states['R'][agent]);r=rh[-1]
        degradation=max(0.,rh[0,0]-r[0])
        probability=math.exp(-max(.001,1-r[0]+2*degradation)*p.get('stability',1.)*(1+duration/40))
        # All agents calling a service at a given time share its health trajectory.
        for tool in t.tools:
            health=self.wave(('tool',tool),started+duration/2)
            probability*=max(.5,1-.10*max(0.,health)-.02*duration/max(1,p.get('tool_quota',60)))
        quality=1-.12*(1-p.get('proficiency',.9))*(t.work/400)
        success=finished and rng.random()<probability*max(.5,quality)
        self.last_reason='' if success else ('execution_timeout' if not finished else 'tool_failure' if t.tools and probability<.9 else 'execution_quality_failure')
        return duration,bool(success)
    def transfer(self,workflow,parent,child,source,target,mb,started,attempt):
        s=self.snapshot;path=s.path(source,target)
        if path is None:self.last_reason='disconnected';return 0.,False
        if not path:self.last_reason='';return 0.,True
        raw=np.asarray([s.states['E'][i][-1] for i in path])
        latency=float(raw[:,0].sum())
        def rate(now):
            return min(s.bandwidth[l]*max(.12,1+.35*self.wave(('link',l),now)) for l in path)/8
        duration,finished=self.integrate(mb,started+latency,rate)
        duration+=latency
        congestion=np.mean([max(0.,-self.wave(('link',l),started+duration/2)) for l in path])
        probability=float(np.prod(1-raw[:,2]))*math.exp(-.025*congestion*len(path)-.002*duration)
        success=finished and self.rng('edge',workflow.id,parent,child,source,target,attempt).random()<probability
        self.last_reason='' if success else 'transfer_failure'
        return duration,bool(success)
