"""
Telegram Bot for Quota Stats.

Provides quota information via Telegram commands instead of the TUI.
Uses the same /v1/quota-stats API endpoint as the quota_viewer.

Commands:
    /start - Welcome message and help
    /quota - Summary of all providers
    /quota <provider> - Detailed view for specific provider
    /refresh - Force refresh quota from API

Setup:
    1. Create a bot via @BotFather on Telegram
    2. Set TELEGRAM_BOT_TOKEN in .env
    3. Get your user ID from @userinfobot
    4. Set TELEGRAM_ALLOWED_USERS in .env (comma-separated IDs)
    5. Run: python -m src.proxy_app.telegram_bot
"""

import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

# Telegram imports
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    from telegram.constants import ParseMode
except ImportError:
    print("Error: python-telegram-bot not installed.")
    print("Run: pip install 'python-telegram-bot>=21.0'")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


def get_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set in environment")
        sys.exit(1)

    # Parse allowed user IDs
    allowed_users_str = os.getenv("TELEGRAM_ALLOWED_USERS", "")
    allowed_users: List[int] = []
    if allowed_users_str:
        try:
            allowed_users = [int(uid.strip()) for uid in allowed_users_str.split(",")]
        except ValueError:
            logger.error(
                "Invalid TELEGRAM_ALLOWED_USERS format. Use comma-separated IDs."
            )
            sys.exit(1)

    # Proxy configuration
    proxy_host = os.getenv("PROXY_HOST", "127.0.0.1")
    proxy_port = int(os.getenv("PROXY_PORT", "8000"))
    proxy_api_key = os.getenv("PROXY_API_KEY", "")
    proxy_scheme = os.getenv(
        "PROXY_SCHEME", ""
    )  # "http" or "https", auto-detect if empty

    return {
        "token": token,
        "allowed_users": allowed_users,
        "proxy_host": proxy_host,
        "proxy_port": proxy_port,
        "proxy_api_key": proxy_api_key,
        "proxy_scheme": proxy_scheme,
    }


CONFIG = get_config()


# =============================================================================
# Formatting Helpers
# =============================================================================


def format_tokens(count: int) -> str:
    """Format token count for display (e.g., 125000 -> 125k)."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.0f}k"
    return str(count)


def format_cost(cost: Optional[float]) -> str:
    """Format cost for display."""
    if cost is None or cost == 0:
        return "-"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def create_progress_bar(percent: Optional[int], width: int = 10) -> str:
    """Create a text-based progress bar using Unicode blocks."""
    if percent is None:
        return "░" * width
    filled = int(percent / 100 * width)
    return "▓" * filled + "░" * (width - filled)


def format_cooldown(seconds: int) -> str:
    """Format cooldown seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins}m {secs}s" if secs > 0 else f"{mins}m"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours}h {mins}m" if mins > 0 else f"{hours}h"


def escape_markdown(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    special_chars = [
        "_",
        "*",
        "[",
        "]",
        "(",
        ")",
        "~",
        "`",
        ">",
        "#",
        "+",
        "-",
        "=",
        "|",
        "{",
        "}",
        ".",
        "!",
    ]
    for char in special_chars:
        text = text.replace(char, f"\\{char}")
    return text


# =============================================================================
# API Client
# =============================================================================


async def fetch_quota_stats(provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch quota stats from the proxy API."""
    host = CONFIG["proxy_host"]
    port = CONFIG["proxy_port"]
    api_key = CONFIG["proxy_api_key"]
    scheme = CONFIG["proxy_scheme"]

    if not scheme:
        if (
            host in ("localhost", "127.0.0.1", "::1")
            or host.startswith("192.168.")
            or host.startswith("10.")
        ):
            scheme = "http"
        else:
            scheme = "https"

    url = f"{scheme}://{host}:{port}/v1/quota-stats"
    if provider:
        url += f"?provider={provider}"

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)

            if response.status_code == 401:
                return {"error": "Authentication failed. Check PROXY_API_KEY."}
            elif response.status_code != 200:
                return {"error": f"HTTP {response.status_code}: {response.text[:100]}"}

            return response.json()

    except httpx.ConnectError:
        return {"error": "Connection failed. Is the proxy running?"}
    except httpx.TimeoutException:
        return {"error": "Request timed out."}
    except Exception as e:
        return {"error": str(e)}


async def post_refresh_action(
    action: str = "reload",
    scope: str = "all",
    provider: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Post a refresh action to the proxy."""
    host = CONFIG["proxy_host"]
    port = CONFIG["proxy_port"]
    api_key = CONFIG["proxy_api_key"]
    scheme = CONFIG["proxy_scheme"]

    if not scheme:
        if (
            host in ("localhost", "127.0.0.1", "::1")
            or host.startswith("192.168.")
            or host.startswith("10.")
        ):
            scheme = "http"
        else:
            scheme = "https"

    url = f"{scheme}://{host}:{port}/v1/quota-stats"

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"action": action, "scope": scope}
    if provider:
        payload["provider"] = provider

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)

            if response.status_code == 401:
                return {"error": "Authentication failed."}
            elif response.status_code != 200:
                return {"error": f"HTTP {response.status_code}"}

            return response.json()

    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Authorization
# =============================================================================


def is_authorized(user_id: int) -> bool:
    """Check if user is authorized to use the bot."""
    allowed = CONFIG["allowed_users"]
    # If no users specified, deny all (security default)
    if not allowed:
        logger.warning(
            f"Unauthorized access attempt from user {user_id} (no allowed users configured)"
        )
        return False
    return user_id in allowed


# =============================================================================
# Message Formatters
# =============================================================================


def format_summary_message(stats: Dict[str, Any]) -> str:
    """Format the quota summary for Telegram."""
    if "error" in stats:
        return f"❌ Error: {stats['error']}"

    providers = stats.get("providers", {})
    if not providers:
        return "📊 No providers configured."

    lines = ["📊 *Quota Summary*", ""]

    for provider, prov_stats in providers.items():
        cred_count = prov_stats.get("credential_count", 0)
        active_count = prov_stats.get("active_count", 0)
        exhausted_count = prov_stats.get("exhausted_count", 0)

        # Status emoji
        if exhausted_count > 0:
            status = "🔴"
        elif active_count < cred_count:
            status = "🟡"
        else:
            status = "🟢"

        # Requests and tokens
        total_requests = prov_stats.get("total_requests", 0)
        tokens = prov_stats.get("tokens", {})
        input_total = tokens.get("input_cached", 0) + tokens.get("input_uncached", 0)
        output = tokens.get("output", 0)
        cost = format_cost(prov_stats.get("approx_cost"))

        lines.append(f"{status} *{provider}*")
        lines.append(f"   Creds: {active_count}/{cred_count} active")

        # Quota groups
        quota_groups = prov_stats.get("quota_groups", {})
        for group_name, group_stats in quota_groups.items():
            total_used = group_stats.get("total_requests_used", 0)
            total_max = group_stats.get("total_requests_max", 0)
            total_pct = group_stats.get("total_remaining_pct")

            bar = create_progress_bar(total_pct)
            pct_str = f"{total_pct}%" if total_pct is not None else "?"

            lines.append(f"   `{group_name}: {total_used}/{total_max} {pct_str}`")
            lines.append(f"   `{bar}`")

        lines.append(
            f"   📈 {total_requests} reqs | {format_tokens(input_total)}/{format_tokens(output)} tok | {cost}"
        )
        lines.append("")

    # Summary
    summary = stats.get("summary", {})
    total_creds = summary.get("total_credentials", 0)
    total_reqs = summary.get("total_requests", 0)
    total_tokens = summary.get("tokens", {})
    total_input = total_tokens.get("input_cached", 0) + total_tokens.get(
        "input_uncached", 0
    )
    total_output = total_tokens.get("output", 0)
    total_cost = format_cost(summary.get("approx_total_cost"))

    lines.append("─" * 30)
    lines.append(f"*Total:* {total_creds} creds | {total_reqs} reqs | {total_cost}")

    return "\n".join(lines)


def format_provider_detail(provider: str, stats: Dict[str, Any]) -> str:
    """Format detailed provider stats for Telegram."""
    if "error" in stats:
        return f"❌ Error: {stats['error']}"

    providers = stats.get("providers", {})
    prov_stats = providers.get(provider)

    if not prov_stats:
        available = ", ".join(providers.keys()) if providers else "none"
        return f"❌ Provider '{provider}' not found.\n\nAvailable: {available}"

    lines = [f"📊 *{provider.title()} Details*", ""]

    credentials = prov_stats.get("credentials", [])

    if not credentials:
        lines.append("_No credentials configured._")
        return "\n".join(lines)

    for idx, cred in enumerate(credentials, 1):
        identifier = cred.get("identifier", f"cred-{idx}")
        email = cred.get("email", identifier)
        tier = cred.get("tier", "")
        status = cred.get("status", "unknown")

        # Status icon
        key_cooldown = cred.get("key_cooldown_remaining")
        if status == "exhausted":
            status_icon = "⛔"
            status_text = "Exhausted"
        elif status == "cooldown" or key_cooldown:
            cd_str = format_cooldown(int(key_cooldown)) if key_cooldown else ""
            status_icon = "⚠️"
            status_text = f"Cooldown {cd_str}"
        else:
            status_icon = "✅"
            status_text = "Active"

        tier_str = f" ({tier})" if tier else ""
        lines.append(f"{status_icon} *\\[{idx}\\] {escape_markdown(email)}{tier_str}*")
        lines.append(f"   Status: {status_text}")

        # Stats
        requests = cred.get("requests", 0)
        tokens = cred.get("tokens", {})
        input_total = tokens.get("input_cached", 0) + tokens.get("input_uncached", 0)
        output = tokens.get("output", 0)
        cost = format_cost(cred.get("approx_cost"))

        lines.append(
            f"   📈 {requests} reqs | {format_tokens(input_total)}/{format_tokens(output)} tok | {cost}"
        )

        # Model groups with quota
        model_groups = cred.get("model_groups", {})
        if model_groups:
            for group_name, group_stats in model_groups.items():
                remaining_pct = group_stats.get("remaining_pct")
                requests_used = group_stats.get("requests_used", 0)
                requests_max = group_stats.get("requests_max")
                is_exhausted = group_stats.get("is_exhausted", False)

                display = (
                    f"{requests_used}/{requests_max}"
                    if requests_max
                    else f"{requests_used}/?"
                )
                bar = create_progress_bar(remaining_pct)
                pct_str = f"{remaining_pct}%" if remaining_pct is not None else "?"

                if is_exhausted:
                    emoji = "🔴"
                elif remaining_pct and remaining_pct < 20:
                    emoji = "🟡"
                else:
                    emoji = "🟢"

                lines.append(f"   {emoji} `{group_name}: {display} {pct_str}`")
                lines.append(f"      `{bar}`")

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Command Handlers
# =============================================================================


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    user = update.effective_user
    if not user or not is_authorized(user.id):
        await update.message.reply_text(
            "⛔ Unauthorized. Your user ID is not in the allowed list."
        )
        return

    welcome = """🤖 *Quota Stats Bot*

Available commands:

/quota \\- Summary of all providers
/quota \\[provider\\] \\- Details for a provider
/refresh \\- Force refresh quota data

Example: `/quota antigravity`
"""
    await update.message.reply_text(welcome, parse_mode=ParseMode.MARKDOWN_V2)


async def quota_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /quota command."""
    user = update.effective_user
    if not user or not is_authorized(user.id):
        await update.message.reply_text("⛔ Unauthorized.")
        return

    # Check if a specific provider was requested
    provider = None
    if context.args:
        provider = context.args[0].lower()

    # Send "loading" message
    loading_msg = await update.message.reply_text("⏳ Fetching quota stats...")

    try:
        stats = await fetch_quota_stats(provider)

        if stats is None:
            stats = {"error": "Failed to fetch stats"}

        if provider:
            message = format_provider_detail(provider, stats)
        else:
            message = format_summary_message(stats)

        # Try MarkdownV2 first, fall back to plain text
        try:
            await loading_msg.edit_text(message, parse_mode=ParseMode.MARKDOWN_V2)
        except Exception:
            # Strip markdown and send as plain text
            plain_message = message.replace("*", "").replace("`", "").replace("\\", "")
            await loading_msg.edit_text(plain_message)

    except Exception as e:
        logger.exception("Error fetching quota stats")
        await loading_msg.edit_text(f"❌ Error: {str(e)}")


async def refresh_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /refresh command."""
    user = update.effective_user
    if not user or not is_authorized(user.id):
        await update.message.reply_text("⛔ Unauthorized.")
        return

    # Check if a specific provider was requested
    provider = None
    scope = "all"
    if context.args:
        provider = context.args[0].lower()
        scope = "provider"

    loading_msg = await update.message.reply_text("🔄 Refreshing quota data...")

    try:
        result = await post_refresh_action("force_refresh", scope, provider)

        if result and "error" in result:
            await loading_msg.edit_text(f"❌ {result['error']}")
            return

        if result and result.get("refresh_result"):
            rr = result["refresh_result"]
            creds = rr.get("credentials_refreshed", 0)
            duration = rr.get("duration_ms", 0)
            await loading_msg.edit_text(
                f"✅ Refreshed {creds} credentials in {duration}ms"
            )
        else:
            await loading_msg.edit_text("✅ Refresh complete")

        # Fetch and show updated stats
        await asyncio.sleep(0.5)
        stats = await fetch_quota_stats(provider)
        if stats is None:
            stats = {"error": "Failed to fetch stats"}
        if provider:
            message = format_provider_detail(provider, stats)
        else:
            message = format_summary_message(stats)

        try:
            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)
        except Exception:
            plain_message = message.replace("*", "").replace("`", "").replace("\\", "")
            await update.message.reply_text(plain_message)

    except Exception as e:
        logger.exception("Error refreshing quota")
        await loading_msg.edit_text(f"❌ Error: {str(e)}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await start_command(update, context)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the Telegram bot."""
    print("=" * 60)
    print("  Telegram Quota Stats Bot")
    print("=" * 60)
    print()

    token = CONFIG["token"]
    allowed_users = CONFIG["allowed_users"]

    if not allowed_users:
        print("⚠️  WARNING: No TELEGRAM_ALLOWED_USERS configured!")
        print("   The bot will reject all requests for security.")
        print("   Set TELEGRAM_ALLOWED_USERS=your_user_id in .env")
        print()

    print(f"Proxy: {CONFIG['proxy_host']}:{CONFIG['proxy_port']}")
    print(f"Allowed users: {allowed_users if allowed_users else 'NONE (all blocked)'}")
    print()
    print("Starting bot...")
    print()

    # Create application
    application = Application.builder().token(token).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("quota", quota_command))
    application.add_handler(CommandHandler("refresh", refresh_command))

    # Run the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
