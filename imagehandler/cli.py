from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .commands import background_app, garment_app, items_app, menu_app, quality_app, sheet_app
from .commands.background import batch_remove_cmd, remove_cmd
from .commands.garment import extract_cmd as garment_extract_cmd
from .commands.items import batch_extract_cmd, extract_cmd
from .commands.quality import batch_judge_cmd, judge_cmd
from .commands.sheet import batch_split_cmd, split_cmd
from .menuitem import MenuItem, register_menu_items

app = typer.Typer(no_args_is_help=True, help="ImageHandler CLI")

register_menu_items(
    app,
    [
        MenuItem("menu", "Interactive selection menu", menu_app),
        MenuItem("bg", "Background removal commands", background_app),
        MenuItem("sheet", "Character sheet splitting commands", sheet_app),
        MenuItem("items", "Item/equipment extraction commands", items_app),
        MenuItem("garment", "Person clothing separation commands", garment_app),
        MenuItem("quality", "Quality judging commands", quality_app),
    ],
)

# Backward-compatible legacy aliases.
app.command("remove-bg", help="Alias of: imagehandler bg remove")(remove_cmd)
app.command("batch-remove-bg", help="Alias of: imagehandler bg batch-remove")(batch_remove_cmd)
app.command("split-sheet", help="Alias of: imagehandler sheet split")(split_cmd)
app.command("batch-split-sheet", help="Alias of: imagehandler sheet batch-split")(batch_split_cmd)
app.command("extract-items", help="Alias of: imagehandler items extract")(extract_cmd)
app.command("batch-extract-items", help="Alias of: imagehandler items batch-extract")(batch_extract_cmd)
app.command("extract-garment", help="Alias of: imagehandler garment extract")(garment_extract_cmd)
app.command("judge", help="Alias of: imagehandler quality judge")(judge_cmd)
app.command("batch-judge", help="Alias of: imagehandler quality batch-judge")(batch_judge_cmd)


if __name__ == "__main__":
    app()
