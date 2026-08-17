#!/usr/bin/env python3

# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""List MJLab environments registered by Robot Learning Lab."""

import argparse

import mjlab.tasks  # noqa: F401
import robot_learning_lab_tasks.tasks.mjlab  # noqa: F401
from mjlab.tasks.registry import list_tasks
from prettytable import PrettyTable


def main() -> None:
    """Print registered MJLab environments, optionally filtered by keyword."""
    parser = argparse.ArgumentParser(description="List MJLab environments.")
    parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter environments.")
    args = parser.parse_args()

    table = PrettyTable(["S. No.", "Task ID"])
    table.title = "Available Environments in MJLab"
    table.align["Task ID"] = "l"

    keyword = args.keyword.lower() if args.keyword else None
    task_ids = [task_id for task_id in list_tasks() if keyword is None or keyword in task_id.lower()]

    for index, task_id in enumerate(task_ids, start=1):
        table.add_row([index, task_id])

    print(table)
    if not task_ids:
        message = "[INFO] No tasks matched"
        if args.keyword:
            message += f" keyword '{args.keyword}'"
        print(message)


if __name__ == "__main__":
    main()
