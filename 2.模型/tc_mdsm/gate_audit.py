"""Validation-only audit of soft training gates versus deployed discrete gates."""
import numpy as np
import torch

def gate_audit(model,examples,limit=18):
    model.eval();groups=[];changes=[];errors={'hard':[],'soft':[]}
    if not model.sparse:return {'applicable':False}
    with torch.no_grad():
        for i in range(min(limit,len(examples))):
            e=examples[i];outputs={}
            forced={(o.task,o.agent) for o in e.observations}
            forced|={(o.target_task,o.target_agent) for o in e.observations if o.kind=='edge'}
            for mode in ('hard','soft'):
                model._continuous_gates=mode=='soft'
                out=model(*e.inputs(),forced_pairs=forced);outputs[mode]=out
                decoded=model.decode(out,*e.inputs())
                maps={'node':{k:j for j,k in enumerate(out['pairs'])},'edge':{k:j for j,k in enumerate(out['edge_keys'])}}
                for o in e.observations:
                    key=(o.task,o.agent) if o.kind=='node' else (o.task,o.target_task,o.agent,o.target_agent)
                    errors[mode].append(abs(float(decoded[o.kind][0][maps[o.kind][key]])-o.duration))
            groups.extend(outputs['hard'].get('gates',torch.empty((0,4))).cpu().tolist())
            for task,ids in outputs['hard']['candidates'].items():
                other=outputs['soft']['candidates'][task]
                changes.append(len(set(ids)^set(other))/max(1,len(set(ids)|set(other))))
    model._continuous_gates=False;model._state_cache=None;model._index_cache=None
    return {'applicable':True,'validation_examples':min(limit,len(examples)),'group_active_fraction':np.mean(groups,axis=0).tolist() if groups else [],'candidate_disagreement':float(np.mean(changes)) if changes else 0.,'observed_mae_s':{k:float(np.mean(v)) if v else None for k,v in errors.items()}}
