from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'2.模型'))
# Reuse compatible scientific dependencies while the isolated GPU wheel is loaded first.
import torch
sys.path.append(str(ROOT/'.venv-tc'/'Lib'/'site-packages'))
from tc_mdsm.data import Foundation
from tc_mdsm.model import Predictor,observed_loss
from tc_mdsm.types import ModelConfig
from tc_mdsm.training import atomic_json
torch.set_num_threads(2);f=Foundation(ROOT/'1.数据');e=f.episode('train',321,agents=100,difficulty='中等')[0]
m=Predictor(ModelConfig()).to('cuda');m.fit_normalizers([e]);opt=torch.optim.Adam(m.parameters(),lr=.001)
times=[]
for k in range(12):
    torch.cuda.synchronize();start=time.perf_counter();m.train();opt.zero_grad();loss,_=observed_loss(m,e);loss.backward();opt.step();torch.cuda.synchronize()
    if k>=2:times.append(time.perf_counter()-start)
result={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'hidden':128,'candidate_budget':32,'mean_train_step_s':sum(times)/len(times),'train_peak_bytes':torch.cuda.max_memory_allocated(),'projected_12000_epoch_s_excluding_validation':12000*sum(times)/len(times)}
del loss;opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();m.eval()
big=f.episode('val',322,agents=1600,difficulty='困难')[0]
with torch.no_grad():pred=m.predict(*big.inputs(),budget=32)
result.update(large_graph_agents=1600,large_graph_peak_bytes=torch.cuda.max_memory_allocated(),large_graph_prediction_s=pred.timing['total'])
atomic_json(ROOT/'6.结果输出'/'2026-09-27_TC-MDSM正式实验'/'GPU实测.json',result);print(json.dumps(result),flush=True)
