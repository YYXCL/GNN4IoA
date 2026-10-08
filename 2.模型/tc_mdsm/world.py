"""第2步的第一个前置：独立动态环境：不读取任何策略身份或预测标签。

积分采用 0.5 秒步长，最后一个不足整步的部分按比例折算。后台服务与工具健康轨迹
在所有方法间是共同的随机过程。execution.py 中的资源预留负责处理已接纳任务之间的竞争。
"""
from __future__ import annotations
import math
import numpy as np
from .data import seed_of

class World:
    """仿真"物理世界"：给定随机种子的确定性执行/传输环境。"""
    def __init__(self,snapshot,seed):
        self.snapshot,self.seed=snapshot,seed
        self.last_reason=''
    def rng(self,*key):
        """按关键字派生一个独立的随机数生成器（保证同一 key 结果可复现）。"""
        return np.random.default_rng(seed_of(self.seed,*key))
    def profile(self,i):
        """取得第 i 个智能体的仿真行为参数。"""
        return self.snapshot.agent_profiles[i] if self.snapshot.agent_profiles else {}
    def wave(self,key,t):
        """公共随机波动过程（两个正弦叠加），用于模拟算力/工具/链路的时变波动。"""
        phase=(seed_of(self.seed,'phase',key)%100000)/100000*2*math.pi
        return math.sin(t/12+phase)+.4*math.sin(t/3.7+phase*2)
    def integrate(self,amount,start,rate,limit=1200.):
        """按 0.5 秒步长对时变速率积分，返回 (耗时, 是否在限时内完成)。"""
        elapsed=0.
        while amount>1e-9 and elapsed<limit:
            speed=max(.01,float(rate(start+elapsed)))
            step=min(.5,amount/speed)
            amount-=speed*step;elapsed+=step
        return max(.001,elapsed),amount<=1e-9
    def execute(self,workflow,task,agent,started,attempt):
        """仿真一次任务执行，返回 (耗时, 是否成功)。"""
        s=self.snapshot;t=workflow.tasks[task];c=s.agent_compute[agent]
        hist=np.asarray(s.states['C'][c]);cur=hist[-1]
        # 算力的短期趋势（最近几步的斜率）
        trend=(cur[0]-hist[max(0,len(hist)-4),0])/max(1,min(3,len(hist)-1))
        p=self.profile(agent);speed=p.get('speed',1.)
        rng=self.rng('node',workflow.id,task,agent,attempt)
        noise=float(rng.lognormal(-.005,.10))     # 对数正态噪声
        def rate(now):
            # 瞬时执行速率 = 基础算力 * 智能体速度 * 算力波动 / 噪声
            base=max(4.,cur[0]+trend*min(20.,max(0.,now)))
            return base*speed*max(.15,1+.30*self.wave(('compute',c),now))/noise
        duration,finished=self.integrate(t.work,started,rate)
        rh=np.asarray(s.states['R'][agent]);r=rh[-1]
        # 可靠性下降会使成功率指数衰减
        degradation=max(0.,rh[0,0]-r[0])
        probability=math.exp(-max(.001,1-r[0]+2*degradation)*p.get('stability',1.)*(1+duration/40))
        # 同一时刻调用同一服务的所有智能体共享该服务的健康轨迹
        for tool in t.tools:
            health=self.wave(('tool',tool),started+duration/2)
            probability*=max(.5,1-.10*max(0.,health)-.02*duration/max(1,p.get('tool_quota',60)))
        quality=1-.12*(1-p.get('proficiency',.9))*(t.work/400)
        success=finished and rng.random()<probability*max(.5,quality)
        self.last_reason='' if success else ('execution_timeout' if not finished else 'tool_failure' if t.tools and probability<.9 else 'execution_quality_failure')
        return duration,bool(success)
    def transfer(self,workflow,parent,child,source,target,mb,started,attempt):
        """仿真一次任务间数据传输，返回 (耗时, 是否成功)。"""
        s=self.snapshot;path=s.path(source,target)
        if path is None:self.last_reason='disconnected';return 0.,False
        if not path:self.last_reason='';return 0.,True    # 同一智能体内部传递无网络开销
        raw=np.asarray([s.states['E'][i][-1] for i in path])
        latency=float(raw[:,0].sum())
        def rate(now):
            # 传输速率取路径上最慢链路的时变带宽（Mbps 转 MB/s 除以 8）
            return min(s.bandwidth[l]*max(.12,1+.35*self.wave(('link',l),now)) for l in path)/8
        duration,finished=self.integrate(mb,started+latency,rate)
        duration+=latency
        # 链路拥塞越大、传输越长，失败概率越高
        congestion=np.mean([max(0.,-self.wave(('link',l),started+duration/2)) for l in path])
        probability=float(np.prod(1-raw[:,2]))*math.exp(-.025*congestion*len(path)-.002*duration)
        success=finished and self.rng('edge',workflow.id,parent,child,source,target,attempt).random()<probability
        self.last_reason='' if success else 'transfer_failure'
        return duration,bool(success)
