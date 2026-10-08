"""内容身份标识：用于防止把不兼容的运行结果静默混在一起。"""
from pathlib import Path
import hashlib
import json


def code_identity():
    """对包内所有 .py 文件取哈希，记录本次运行对应的代码版本。"""
    root=Path(__file__).resolve().parent
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}


def digest(value):
    """对任意 JSON 可序列化的值取稳定哈希（键排序，保证可复现）。"""
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def model_identity(model):
    """对模型所有参数取哈希，标识一个具体的模型权重。"""
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
