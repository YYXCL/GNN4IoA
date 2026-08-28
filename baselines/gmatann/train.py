"""在 GNN4IoA 同一时序划分上训练 GMATANN/GCAN 单子任务基线。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn

from gcan_baseline import GCANConfig, GraphConvolutionalAssignmentNetwork, load_cases


def accuracy(model, cases) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for case in cases:
            correct += int(int(model(case)["candidate_logits"].argmax()) == int(case.label))
    return correct / max(1, len(cases))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="GMATANN/GCAN")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(1)

    train_cases = load_cases(args.dataset / "train.jsonl")
    val_cases = load_cases(args.dataset / "val.jsonl")
    config = GCANConfig(numeric_dim=train_cases[0].agent_numeric.shape[1])
    model = GraphConvolutionalAssignmentNetwork(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    classification = nn.CrossEntropyLoss()
    state_loss = nn.BCELoss()
    best_accuracy = -1.0
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_cases)
        loss_sum = 0.0
        for case in train_cases:
            output = model(case)
            loss = classification(output["candidate_logits"].unsqueeze(0), case.label.reshape(1))
            loss = loss + 0.20 * state_loss(output["state_predictions"], case.candidate_targets)
            teacher = torch.softmax(case.teacher_scores / 0.15, dim=0)
            loss = loss + 0.10 * nn.functional.kl_div(
                torch.log_softmax(output["candidate_logits"], dim=0), teacher, reduction="batchmean",
            )
            loss = loss + 0.03 * output["kl_loss"] + 0.05 * output["reconstruction_loss"]
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
        validation_accuracy = accuracy(model, val_cases)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(train_cases),
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save({"config": config.to_dict(), "model": model.state_dict(), "name": args.name}, args.output)
    (args.output.parent / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"checkpoint={args.output}")


if __name__ == "__main__":
    main()
