from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat_app.api import create_app


app = create_app()


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=7861)


if __name__ == "__main__":
    main()
