"""执行数据生成、训练、动态仿真评估与四幅论文图绘制。"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import asdict
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("TC-MDSM", "GMATANN", "HM-MATAS", "Rule-Latency", "Full-State", "StaticGraph")


def integers(text):
    try:
        values = [int(item) for item in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("请输入逗号分隔的整数") from exc
    if not values:
        raise argparse.ArgumentTypeError("列表不能为空")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="all", choices=("all", "generate", "train", "evaluate", "report"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--counts", type=integers, default=[12000, 1500, 1500])
    parser.add_argument("--seeds", type=integers, default=[7, 17, 29, 43, 61])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--instances", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--planning-batch-size", type=int, default=4)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="训练设备")
    parser.add_argument("--evaluation-device", default="cpu", choices=("cpu", "cuda"), help="各方法和各条件统一的评估设备")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--scale-agents", type=integers, default=[50, 100, 200, 400, 800, 1600])
    parser.add_argument("--concurrency", type=integers, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--deadline-policy", default="hard", choices=("hard", "evaluate"))
    parser.add_argument("--solver-start", default="serial", choices=("serial", "reject"))
    parser.add_argument("--execution-policy", default="ready", choices=("ready", "planned"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--describe", action="store_true", help="只查看运行设置，不生成数据或启动训练")
    args = parser.parse_args()
    if len(args.counts) != 3 or min(args.counts) <= 0:
        parser.error("counts 必须为训练、验证、测试三个正数")
    if min(args.epochs, args.instances, args.seconds, args.candidates, args.planning_batch_size, args.threads) <= 0 or args.validation_limit < 0:
        parser.error("轮数、批数、规划预算、候选数、规划批大小和线程数须为正")
    if min(args.scale_agents + args.concurrency) <= 0:
        parser.error("智能体数量和并发工作流数量须为正")
    if any(len(set(values)) != len(values) for values in (args.seeds, args.scale_agents, args.concurrency)):
        parser.error("种子、智能体数量和并发工作流数量不能包含重复值")
    if args.resume and args.output is None:
        parser.error("续跑时请用 --output 指定原实验目录")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d_%H-%M-%S")
    out = (args.output or ROOT / "4.结果输出" / f"{stamp}_论文实验").resolve()
    if args.describe:
        print(json.dumps({**vars(args), "output": str(out), "samples": str(args.samples) if args.samples else None}, ensure_ascii=False, indent=2))
        return

    from tc_mdsm_v3_runner import Cases, generate, evaluate, train, save_json, ModelConfig, torch
    from tc_mdsm_v3_report import report
    if args.stage == "report":
        report(out, paper_style=True)
        return
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if (device == "cuda" or args.evaluation_device == "cuda") and not torch.cuda.is_available():
        parser.error("当前环境没有可用CUDA，请选择cpu")
    torch.set_num_threads(args.threads)
    config = ModelConfig(candidate_budgets=(args.candidates,), planning_batch_size=args.planning_batch_size,
                         deadline_policy=args.deadline_policy, solver_start=args.solver_start,
                         execution_policy=args.execution_policy)
    methods = [args.method] if args.method else list(METHODS)
    sample_path = (args.samples or out / "数据" / "samples.sqlite").resolve()
    parameters = dict(counts=args.counts, seeds=args.seeds, epochs=args.epochs, instances=args.instances,
                      seconds=args.seconds, validation_limit=args.validation_limit, device=device,
                      evaluation_device=args.evaluation_device, threads=args.threads,
                      model=asdict(config), methods=methods, scale_agents=args.scale_agents,
                      concurrency=args.concurrency, topology="agent_complete_compute_leaf",
                      duration_loss="log_ratio", samples_source=str(sample_path),
                      data_source="world_simulation", figure_style="english_no_error_bars")
    parameter_path = out / "实验参数.json"
    if parameter_path.exists():
        existing = json.loads(parameter_path.read_text(encoding="utf-8"))
        if existing != json.loads(json.dumps(parameters)):
            parser.error("运行设置与该目录已保存设置不同，请使用新的输出目录，避免混合结果")
        if args.stage == "all" and not args.resume:
            parser.error("目录已存在；续跑请加 --resume，重新训练请使用新目录")
    elif args.stage not in ("all", "generate"):
        parser.error("请先运行all或generate生成实验参数与样本")
    elif args.resume:
        parser.error("指定目录没有可续跑的实验参数")
    out.mkdir(parents=True, exist_ok=True)
    save_json(parameter_path, parameters)
    print(f"输出目录：{out}\n训练设备：{device}；评估设备：{args.evaluation_device}", flush=True)
    if args.stage in ("all", "generate") and args.samples is None:
        generate(out / "数据", args.counts)
    if args.stage in ("all", "train"):
        cases, validation = Cases(sample_path, "train"), Cases(sample_path, "val")
        try:
            if len(cases) != args.counts[0] or len(validation) != args.counts[1]:
                raise ValueError("样本数据库的训练/验证数量与运行设置不一致")
            for seed in args.seeds:
                for method in methods:
                    train(cases, validation, out / "训练" / f"seed_{seed}" / method.lower().replace("-", "_"),
                          method, config, seed=seed, epochs=args.epochs, device=device, resume=args.resume,
                          validation_limit=args.validation_limit, time_limit=args.seconds)
        finally:
            cases.conn.close(); validation.conn.close()
    if args.stage in ("all", "evaluate"):
        if device == "cuda":torch.cuda.empty_cache()
        evaluate(out, args.seeds, args.instances, args.evaluation_device, args.seconds,
                 args.candidates, args.planning_batch_size, methods, args.scale_agents, args.concurrency,
                 args.deadline_policy, args.solver_start, args.execution_policy)
    if args.stage == "all":
        report(out, paper_style=True)
    print("本阶段已完成。", flush=True)


if __name__ == "__main__":
    main()
