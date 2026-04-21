from warnings import filterwarnings

from dishka import make_container
from mmar_mimpl import init_logger
from telegram.warnings import PTBUserWarning

from frontend_telegram.config import Config
from frontend_telegram.frontend_telegram import FrontendTelegram
from frontend_telegram.ioc import IOC, IOCLocal

# remove per_message=False CallbackQueryHandler warning
filterwarnings(action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning)


def serve(config: Config | None = None) -> None:
    """Run the frontend Telegram bot application.

    Args:
        config: Optional configuration override. If None, loads from environment.
    """
    container = make_container(IOC(), IOCLocal())
    frontend_telegram = container.get(FrontendTelegram)

    # Initialize logger with config from container
    cfg = container.get(Config)
    init_logger(cfg.logger.level)

    try:
        frontend_telegram.run()
    finally:
        container.close()


if __name__ == "__main__":
    serve()
