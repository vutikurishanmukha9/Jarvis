"""
Omnichannel Gateway Package for J.A.R.V.I.S.
Provides seamless remote messaging bridges (Telegram & Discord) for mobile and remote control.
"""

from .telegram_bridge import TelegramGateway, get_telegram_gateway

__all__ = ["TelegramGateway", "get_telegram_gateway"]
