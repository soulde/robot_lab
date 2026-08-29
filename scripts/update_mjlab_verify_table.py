#!/usr/bin/env python3
"""Regenerate the MJLab verification table in robot_lab README.md.

Reads logs/mjlab_ppo/status.tsv (TSV: timestamp, event, task) and rewrites
the section between the MJLAB-VERIFY markers.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
STATUS = ROOT / "logs/mjlab_ppo/status.tsv"
LOGDIR = ROOT / "logs/mjlab_ppo"
START = "<!-- MJLAB-VERIFY-START -->"
END = "<!-- MJLAB-VERIFY-END -->"


def failure_reason(task: str) -> str:
    log = LOGDIR / f"{task}.log"
    if not log.exists():
        return "训练失败"
    for line in reversed(log.read_text(errors="replace").splitlines()):
        line = line.strip()
        if line.startswith(("RuntimeError:", "ValueError:", "AssertionError:", "KeyError:")):
            return f"失败：{line[:80]}"
    return "失败，见 logs/mjlab_ppo/"


def main() -> int:
    status: dict[str, str] = {}
    if STATUS.exists():
        for line in STATUS.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and parts[1] in {"START", "PASS", "FAIL"}:
                status[parts[2]] = parts[1]
    tasks = sorted(
        t
        for t, s in status.items()
        if t.startswith("RobotLab-MJLab-Velocity")
    )
    stopped = (ROOT / "logs/mjlab_ppo/STOP").exists()

    def row(task: str) -> str:
        s = status.get(task)
        if s == "PASS":
            return f"| {task} | ✅ |"
        if s == "FAIL":
            return f"| {task} | ❌ {failure_reason(task)} |"
        if s == "START":
            if stopped:
                return f"| {task} | ⏸️ 被 8 点截止中断 |"
            return f"| {task} | 🏃 训练中 |"
        return f"| {task} | ⬜ |"

    lines = [
        "## MJLab 环境验证",
        "",
        "在独立 mjlab 虚拟环境（`~/mjlab/.venv`）中，用 mjlab 内置 PPO 逐一训练",
        "robot_lab 的全部 MJLab velocity 任务（4096 并行环境，tensorboard 记录，",
        "默认迭代数）。AMP 任务需要 GMR 动作数据（本机无 `~/GMR-private`），不在",
        "内置 PPO 验证范围内。由 `scripts/mjlab_ppo_verify.sh` 顺序执行，本表由",
        "`scripts/update_mjlab_verify_table.py` 依据 `logs/mjlab_ppo/status.tsv` 自动更新。",
        "",
        "| 任务 | 状态 |",
        "|------|------|",
    ]
    lines += [row(t) for t in tasks]
    lines += [
        "",
        "环境基础验证：mjlab venv 安装（editable + mjlab extra）✅、任务注册 ✅、",
        "rll_rl 单元测试 50 passed ✅（2026-08-29）。",
        "",
    ]

    text = README.read_text()
    if START not in text or END not in text:
        print(f"markers missing in {README}", file=sys.stderr)
        return 1
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    README.write_text(head + START + "\n" + "\n".join(lines) + "\n" + END + tail)
    print(f"updated {len(tasks)} task rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
