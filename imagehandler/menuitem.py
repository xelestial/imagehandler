from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import typer


@dataclass(frozen=True)
class MenuItem:
    """Small descriptor used to register a CLI menu group."""

    name: str
    help: str
    app: typer.Typer


def register_menu_items(root: typer.Typer, items: list[MenuItem]) -> None:
    for item in items:
        root.add_typer(item.app, name=item.name, help=item.help)
