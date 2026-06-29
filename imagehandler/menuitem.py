from __future__ import annotations

from dataclasses import dataclass

import typer


@dataclass(frozen=True)
class MenuItem:
    name: str
    help: str
    app: typer.Typer


def register_menu_items(root: typer.Typer, items: list[MenuItem]) -> None:
    for item in items:
        root.add_typer(item.app, name=item.name, help=item.help)
