from __future__ import annotations

from src.config import settings
from ui.gradio_app import build_app


def main() -> None:
    app = build_app()
    app.queue(default_concurrency_limit=1).launch(
        share=settings.share,
        show_error=settings.debug,
        allowed_paths=[str(settings.output_root)],
    )


if __name__ == "__main__":
    main()
