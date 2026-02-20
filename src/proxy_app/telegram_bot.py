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
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

# Telegram imports
try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import (
        Application,
        CallbackQueryHandler,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
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

# Silence httpx INFO logs (noisy getUpdates polling)
logging.getLogger("httpx").setLevel(logging.WARNING)


# =============================================================================
# Session Management (In-Memory)
# =============================================================================


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    """A chat session with message history."""

    id: str
    name: str
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the session."""
        self.messages.append(Message(role=role, content=content))
        self.updated_at = time.time()

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Get messages in OpenAI API format."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []
        self.updated_at = time.time()


@dataclass
class UserState:
    """State for a single user."""

    user_id: int
    sessions: Dict[str, Session] = field(default_factory=dict)
    current_session_id: Optional[str] = None
    selected_model: Optional[str] = None

    def get_or_create_session(self) -> Session:
        """Get current session or create a new one."""
        if self.current_session_id and self.current_session_id in self.sessions:
            return self.sessions[self.current_session_id]
        return self.create_new_session()

    def create_new_session(self, name: Optional[str] = None) -> Session:
        """Create a new session and set it as current."""
        session_id = str(uuid.uuid4())[:8]
        session_name = name or f"Chat {len(self.sessions) + 1}"
        session = Session(id=session_id, name=session_name)
        self.sessions[session_id] = session
        self.current_session_id = session_id
        return session

    def switch_session(self, session_id: str) -> Optional[Session]:
        """Switch to a different session."""
        if session_id in self.sessions:
            self.current_session_id = session_id
            return self.sessions[session_id]
        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.current_session_id == session_id:
                self.current_session_id = None
            return True
        return False


# Global user state storage
USER_STATES: Dict[int, UserState] = {}


def get_user_state(user_id: int) -> UserState:
    """Get or create user state."""
    if user_id not in USER_STATES:
        USER_STATES[user_id] = UserState(user_id=user_id)
    return USER_STATES[user_id]


# =============================================================================
# System Prompt Loading
# =============================================================================

SYSTEM_PROMPT: Optional[str] = None


def load_system_prompt() -> str:
    """Load system prompt from prompts/generic_prompt.md."""
    global SYSTEM_PROMPT
    if SYSTEM_PROMPT is not None:
        return SYSTEM_PROMPT

    # Try multiple paths
    paths_to_try = [
        Path(__file__).parent.parent.parent / "prompts" / "generic_prompt.md",
        Path("prompts/generic_prompt.md"),
        Path(__file__).parent / "prompts" / "generic_prompt.md",
    ]

    for path in paths_to_try:
        if path.exists():
            SYSTEM_PROMPT = path.read_text(encoding="utf-8")
            logger.info(f"Loaded system prompt from {path}")
            return SYSTEM_PROMPT

    logger.warning("System prompt not found, using default")
    SYSTEM_PROMPT = "You are a helpful AI assistant."
    return SYSTEM_PROMPT


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


def format_reset_time(iso_time: Optional[str]) -> str:
    """Format ISO time string for display."""
    if not iso_time:
        return ""
    try:
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        # Convert to local time
        local_dt = dt.astimezone()
        return local_dt.strftime("%b %d %H:%M")
    except (ValueError, AttributeError):
        return iso_time[:16] if iso_time else ""


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


TELEGRAM_MAX_MESSAGE_LENGTH = 4096


def chunk_message(
    text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH
) -> List[str]:
    """Split message into chunks that fit Telegram's limit, splitting on newlines."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    current_chunk = ""

    for line in text.split("\n"):
        line_with_newline = line + "\n"
        if len(current_chunk) + len(line_with_newline) > max_length:
            if current_chunk:
                chunks.append(current_chunk.rstrip("\n"))
            current_chunk = line_with_newline
        else:
            current_chunk += line_with_newline

    if current_chunk:
        chunks.append(current_chunk.rstrip("\n"))

    return chunks


# =============================================================================
# Format Adapters (Modular Usage Manager -> Legacy Format)
# =============================================================================


def get_primary_window(windows: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get the primary window (daily or api_authoritative) from windows dict."""
    if not windows:
        return None
    return (
        windows.get("daily")
        or windows.get("api_authoritative")
        or next(iter(windows.values()), None)
    )


def get_window_remaining_pct(window: Optional[Dict[str, Any]]) -> Optional[int]:
    """Calculate remaining percentage from a window."""
    if not window:
        return None

    remaining = window.get("remaining")
    limit = window.get("limit")

    if remaining is not None and limit and limit > 0:
        return int((remaining / limit) * 100)

    if limit and limit > 0:
        request_count = window.get("request_count", 0)
        return max(0, int(((limit - request_count) / limit) * 100))

    return None


def format_timestamp_to_iso(timestamp: Optional[float]) -> Optional[str]:
    """Convert Unix timestamp to ISO format string."""
    if not timestamp:
        return None
    try:
        from datetime import timezone

        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
    except Exception:
        return None


def adapt_stats_response(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt modular usage manager stats to the legacy format expected by formatting functions.

    Transforms:
    - credentials: dict -> array
    - group_usage with windows -> model_groups with computed fields
    - Aggregates quota_groups at provider level
    """
    if "error" in stats:
        return stats

    adapted = {
        "providers": {},
        "summary": stats.get("summary", {}),
        "timestamp": stats.get("timestamp"),
        "data_source": stats.get("data_source"),
    }

    for provider_name, prov_stats in stats.get("providers", {}).items():
        credentials_dict = prov_stats.get("credentials", {})

        # Handle both dict and array format
        if isinstance(credentials_dict, list):
            # Already an array - use legacy format
            adapted["providers"][provider_name] = prov_stats
            continue

        # Convert credentials dict to array and adapt format
        adapted_credentials = []
        quota_groups_aggregated: Dict[str, Dict[str, Any]] = {}

        for stable_id, cred in credentials_dict.items():
            # Build adapted credential
            totals = cred.get("totals", {})
            adapted_cred = {
                "identifier": cred.get("identifier", stable_id),
                "email": cred.get("email"),
                "tier": cred.get("tier"),
                "status": cred.get("status", "unknown"),
                "requests": totals.get("request_count", 0),
                "tokens": {
                    "input_cached": totals.get("prompt_tokens_cache_read", 0),
                    "input_uncached": totals.get("prompt_tokens", 0),
                    "output": totals.get("output_tokens", 0),
                },
                "approx_cost": totals.get("approx_cost"),
                "model_groups": {},
            }

            # Convert group_usage to model_groups format
            for group_name, group_usage in cred.get("group_usage", {}).items():
                window = get_primary_window(group_usage.get("windows", {}))
                remaining_pct = get_window_remaining_pct(window)

                requests_used = window.get("request_count", 0) if window else 0
                requests_max = window.get("limit") if window else None
                reset_at = window.get("reset_at") if window else None

                is_exhausted = group_usage.get("fair_cycle_exhausted", False)

                adapted_cred["model_groups"][group_name] = {
                    "remaining_pct": remaining_pct,
                    "requests_used": requests_used,
                    "requests_max": requests_max,
                    "is_exhausted": is_exhausted,
                    "reset_time_iso": format_timestamp_to_iso(reset_at)
                    if reset_at
                    else None,
                }

                # Aggregate for provider-level quota_groups
                if group_name not in quota_groups_aggregated:
                    quota_groups_aggregated[group_name] = {
                        "total_requests_used": 0,
                        "total_requests_max": 0,
                        "remaining_pcts": [],
                        "earliest_reset": None,
                    }

                agg = quota_groups_aggregated[group_name]
                agg["total_requests_used"] += requests_used
                if requests_max:
                    agg["total_requests_max"] += requests_max
                if remaining_pct is not None:
                    agg["remaining_pcts"].append(remaining_pct)
                if reset_at:
                    if (
                        agg["earliest_reset"] is None
                        or reset_at < agg["earliest_reset"]
                    ):
                        agg["earliest_reset"] = reset_at

            adapted_credentials.append(adapted_cred)

        # Build provider-level quota_groups
        quota_groups = {}
        for group_name, agg in quota_groups_aggregated.items():
            total_pct = None
            if agg["total_requests_max"] > 0:
                remaining = agg["total_requests_max"] - agg["total_requests_used"]
                total_pct = max(0, int((remaining / agg["total_requests_max"]) * 100))
            elif agg["remaining_pcts"]:
                total_pct = int(sum(agg["remaining_pcts"]) / len(agg["remaining_pcts"]))

            quota_groups[group_name] = {
                "total_requests_used": agg["total_requests_used"],
                "total_requests_max": agg["total_requests_max"],
                "total_remaining_pct": total_pct,
                "next_reset_time_iso": format_timestamp_to_iso(agg["earliest_reset"])
                if agg["earliest_reset"]
                else None,
            }

        adapted["providers"][provider_name] = {
            "credential_count": prov_stats.get("credential_count", 0),
            "active_count": prov_stats.get("active_count", 0),
            "exhausted_count": prov_stats.get("exhausted_count", 0),
            "total_requests": prov_stats.get("total_requests", 0),
            "tokens": prov_stats.get("tokens", {}),
            "approx_cost": prov_stats.get("approx_cost"),
            "credentials": adapted_credentials,
            "quota_groups": quota_groups,
        }

    return adapted


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

            # Adapt new modular format to legacy format
            return adapt_stats_response(response.json())

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

            # Adapt new modular format to legacy format
            return adapt_stats_response(response.json())

    except Exception as e:
        return {"error": str(e)}


def get_proxy_base_url() -> str:
    host = CONFIG["proxy_host"]
    port = CONFIG["proxy_port"]
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

    return f"{scheme}://{host}:{port}"


def get_auth_headers() -> Dict[str, str]:
    api_key = CONFIG["proxy_api_key"]
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


async def fetch_models() -> Optional[Dict[str, Any]]:
    base_url = get_proxy_base_url()
    url = f"{base_url}/v1/models"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=get_auth_headers())

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


async def send_chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    stream: bool = True,
) -> Any:
    base_url = get_proxy_base_url()
    url = f"{base_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    headers = get_auth_headers()
    headers["Content-Type"] = "application/json"

    if stream:
        return httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))
    else:
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(url, headers=headers, json=payload)

                if response.status_code == 401:
                    return {"error": "Authentication failed."}
                elif response.status_code != 200:
                    return {
                        "error": f"HTTP {response.status_code}: {response.text[:200]}"
                    }

                return response.json()

        except httpx.ConnectError:
            return {"error": "Connection failed. Is the proxy running?"}
        except httpx.TimeoutException:
            return {"error": "Request timed out."}
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
            # Get earliest reset time across all credentials in this group
            reset_time = format_reset_time(group_stats.get("next_reset_time_iso"))

            bar = create_progress_bar(total_pct)
            pct_str = f"{total_pct}%" if total_pct is not None else "?"

            lines.append(f"   `{group_name}: {total_used}/{total_max} {pct_str}`")
            # Show reset time if available
            reset_str = f" ⏰ {reset_time}" if reset_time else ""
            lines.append(f"   `{bar}`{reset_str}")

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
        email = cred.get("email") or identifier
        tier = cred.get("tier") or ""
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
                reset_time = format_reset_time(group_stats.get("reset_time_iso"))

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
                # Show reset time if available
                reset_str = f" ⏰ {reset_time}" if reset_time else ""
                lines.append(f"      `{bar}`{reset_str}")

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

*Quota Commands:*
/quota \\\\- Summary of all providers
/quota \\\\[provider\\\\] \\\\- Details for a provider
/refresh \\\\- Force refresh quota data

*Chat Commands:*
/models \\\\- List available models
/model \\\\[name\\\\] \\\\- View or set model
/new \\\\[name\\\\] \\\\- Start new chat session
/sessions \\\\- List your sessions
/session \\\\[id\\\\] \\\\- Switch session
/delete \\\\[id\\\\] \\\\- Delete a session
/clear \\\\- Clear current session

*Account Management:*
/newaccount \\\\- Create a new GitLab Duo trial account

Just send a message to chat with the LLM\\!
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
            keyboard = [
                [
                    InlineKeyboardButton(
                        "🔄 Refresh", callback_data=f"refresh:{provider}"
                    )
                ],
                [InlineKeyboardButton("📊 Summary", callback_data="view:summary")],
            ]
        else:
            message = format_summary_message(stats)
            keyboard = [
                [InlineKeyboardButton("🔄 Refresh All", callback_data="refresh:all")]
            ]
            # Add provider-specific buttons for antigravity
            providers = stats.get("providers", {})
            if "antigravity" in providers:
                keyboard.append(
                    [
                        InlineKeyboardButton(
                            "🔄 Refresh Antigravity",
                            callback_data="refresh:antigravity",
                        ),
                        InlineKeyboardButton(
                            "📋 Antigravity", callback_data="view:antigravity"
                        ),
                    ]
                )

        reply_markup = InlineKeyboardMarkup(keyboard)
        chunks = chunk_message(message)

        try:
            await loading_msg.edit_text(
                chunks[0], parse_mode=ParseMode.MARKDOWN_V2, reply_markup=reply_markup
            )
            for chunk in chunks[1:]:
                await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN_V2)
        except Exception:
            plain_chunks = chunk_message(
                message.replace("*", "").replace("`", "").replace("\\", "")
            )
            await loading_msg.edit_text(plain_chunks[0], reply_markup=reply_markup)
            for chunk in plain_chunks[1:]:
                await update.message.reply_text(chunk)

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

        chunks = chunk_message(message)

        try:
            await update.message.reply_text(chunks[0], parse_mode=ParseMode.MARKDOWN_V2)
            for chunk in chunks[1:]:
                await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN_V2)
        except Exception:
            plain_chunks = chunk_message(
                message.replace("*", "").replace("`", "").replace("\\", "")
            )
            await update.message.reply_text(plain_chunks[0])
            for chunk in plain_chunks[1:]:
                await update.message.reply_text(chunk)

    except Exception as e:
        logger.exception("Error refreshing quota")
        await loading_msg.edit_text(f"❌ Error: {str(e)}")


async def refresh_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle refresh button callback."""
    query = update.callback_query
    if query is None:
        return

    await query.answer()

    user = update.effective_user
    if not user or not is_authorized(user.id):
        await query.edit_message_text("⛔ Unauthorized.")
        return

    # Parse callback data: "refresh:<provider>" or "refresh:all"
    data = query.data or ""
    if not data.startswith("refresh:"):
        return

    provider = data[8:]  # Remove "refresh:" prefix
    if provider == "all":
        provider = None
        scope = "all"
    else:
        scope = "provider"

    # Update message to show loading
    await query.edit_message_text("🔄 Refreshing quota data...")

    try:
        result = await post_refresh_action("force_refresh", scope, provider)

        if result and "error" in result:
            await query.edit_message_text(f"❌ {result['error']}")
            return

        refresh_info = ""
        if result and result.get("refresh_result"):
            rr = result["refresh_result"]
            creds = rr.get("credentials_refreshed", 0)
            duration = rr.get("duration_ms", 0)
            refresh_info = f"✅ Refreshed {creds} credentials in {duration}ms\n\n"

        # Fetch and show updated stats
        await asyncio.sleep(0.5)
        stats = await fetch_quota_stats(provider)
        if stats is None:
            stats = {"error": "Failed to fetch stats"}

        if provider:
            message = refresh_info + format_provider_detail(provider, stats)
            keyboard = [
                [
                    InlineKeyboardButton(
                        "🔄 Refresh", callback_data=f"refresh:{provider}"
                    )
                ],
                [InlineKeyboardButton("📊 Summary", callback_data="view:summary")],
            ]
        else:
            message = refresh_info + format_summary_message(stats)
            keyboard = [
                [InlineKeyboardButton("🔄 Refresh All", callback_data="refresh:all")]
            ]
            # Add provider-specific buttons
            providers = stats.get("providers", {})
            if "antigravity" in providers:
                keyboard.append(
                    [
                        InlineKeyboardButton(
                            "🔄 Refresh Antigravity",
                            callback_data="refresh:antigravity",
                        ),
                        InlineKeyboardButton(
                            "📋 Antigravity", callback_data="view:antigravity"
                        ),
                    ]
                )

        reply_markup = InlineKeyboardMarkup(keyboard)

        chunks = chunk_message(message)
        try:
            await query.edit_message_text(
                chunks[0], parse_mode=ParseMode.MARKDOWN_V2, reply_markup=reply_markup
            )
            # Send additional chunks without buttons
            if query.message:
                for chunk in chunks[1:]:
                    await query.message.reply_text(
                        chunk, parse_mode=ParseMode.MARKDOWN_V2
                    )
        except Exception:
            plain_message = message.replace("*", "").replace("`", "").replace("\\", "")
            plain_chunks = chunk_message(plain_message)
            await query.edit_message_text(plain_chunks[0], reply_markup=reply_markup)
            if query.message:
                for chunk in plain_chunks[1:]:
                    await query.message.reply_text(chunk)

    except Exception as e:
        logger.exception("Error in refresh callback")
        await query.edit_message_text(f"❌ Error: {str(e)}")


async def view_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle view button callback for navigating between quota views."""
    query = update.callback_query
    if query is None:
        return

    await query.answer()

    user = update.effective_user
    if not user or not is_authorized(user.id):
        await query.edit_message_text("⛔ Unauthorized.")
        return

    # Parse callback data: "view:<provider>" or "view:summary"
    data = query.data or ""
    if not data.startswith("view:"):
        return

    view_target = data[5:]  # Remove "view:" prefix
    provider = None if view_target == "summary" else view_target

    # Update message to show loading
    await query.edit_message_text("⏳ Fetching quota stats...")

    try:
        stats = await fetch_quota_stats(provider)
        if stats is None:
            stats = {"error": "Failed to fetch stats"}

        if provider:
            message = format_provider_detail(provider, stats)
            keyboard = [
                [
                    InlineKeyboardButton(
                        "🔄 Refresh", callback_data=f"refresh:{provider}"
                    )
                ],
                [InlineKeyboardButton("📊 Summary", callback_data="view:summary")],
            ]
        else:
            message = format_summary_message(stats)
            keyboard = [
                [InlineKeyboardButton("🔄 Refresh All", callback_data="refresh:all")]
            ]
            # Add provider-specific buttons
            providers = stats.get("providers", {})
            if "antigravity" in providers:
                keyboard.append(
                    [
                        InlineKeyboardButton(
                            "🔄 Refresh Antigravity",
                            callback_data="refresh:antigravity",
                        ),
                        InlineKeyboardButton(
                            "📋 Antigravity", callback_data="view:antigravity"
                        ),
                    ]
                )

        reply_markup = InlineKeyboardMarkup(keyboard)

        chunks = chunk_message(message)
        try:
            await query.edit_message_text(
                chunks[0], parse_mode=ParseMode.MARKDOWN_V2, reply_markup=reply_markup
            )
            if query.message:
                for chunk in chunks[1:]:
                    await query.message.reply_text(
                        chunk, parse_mode=ParseMode.MARKDOWN_V2
                    )
        except Exception:
            plain_message = message.replace("*", "").replace("`", "").replace("\\", "")
            plain_chunks = chunk_message(plain_message)
            await query.edit_message_text(plain_chunks[0], reply_markup=reply_markup)
            if query.message:
                for chunk in plain_chunks[1:]:
                    await query.message.reply_text(chunk)

    except Exception as e:
        logger.exception("Error in view callback")
        await query.edit_message_text(f"❌ Error: {str(e)}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await start_command(update, context)


# =============================================================================
# LLM Chat Commands
# =============================================================================


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not is_authorized(user.id):
        if update.message:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    if update.message is None:
        return

    loading_msg = await update.message.reply_text("🔍 Fetching available models...")

    result = await fetch_models()

    if result is None:
        await loading_msg.edit_text("❌ Failed to fetch models.")
        return

    if "error" in result:
        await loading_msg.edit_text(f"❌ {result['error']}")
        return

    models_data = result.get("data", [])
    if not models_data:
        await loading_msg.edit_text("No models available.")
        return

    model_ids = sorted([m.get("id", "unknown") for m in models_data])

    lines = ["*Available Models:*\n"]
    for model_id in model_ids:
        lines.append(f"• `{model_id}`")

    lines.append(f"\n_Total: {len(model_ids)} models_")
    lines.append("\nUse `/model <name>` to select a model.")

    text = "\n".join(lines)

    chunks = chunk_message(text)
    await loading_msg.edit_text(chunks[0], parse_mode="Markdown")
    for chunk in chunks[1:]:
        await update.message.reply_text(chunk, parse_mode="Markdown")


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not is_authorized(user.id):
        if update.message:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    if update.message is None:
        return

    user_state = get_user_state(user.id)
    args = context.args or []

    if not args:
        current = user_state.selected_model or "Not selected"
        await update.message.reply_text(
            f"*Current model:* `{current}`\n\n"
            "Use `/model <name>` to select a model.\n"
            "Use `/models` to see available models.",
            parse_mode="Markdown",
        )
        return

    model_name = args[0]
    user_state.selected_model = model_name
    await update.message.reply_text(
        f"✅ Model set to: `{model_name}`", parse_mode="Markdown"
    )


async def new_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not is_authorized(user.id):
        if update.message:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    if update.message is None:
        return

    user_state = get_user_state(user.id)
    args = context.args or []
    name = " ".join(args) if args else None

    session = user_state.create_new_session(name)
    await update.message.reply_text(
        f"✅ New session created: *{session.name}* (`{session.id}`)",
        parse_mode="Markdown",
    )


async def sessions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not is_authorized(user.id):
        if update.message:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    if update.message is None:
        return

    user_state = get_user_state(user.id)

    if not user_state.sessions:
        await update.message.reply_text(
            "No sessions yet. Start chatting or use `/new` to create one.",
            parse_mode="Markdown",
        )
        return

    lines = ["*Your Sessions:*\n"]
    for sid, session in user_state.sessions.items():
        is_current = "→ " if sid == user_state.current_session_id else "  "
        msg_count = len(session.messages)
        lines.append(f"{is_current}`{sid}` - {session.name} ({msg_count} msgs)")

    lines.append("\nUse `/session <id>` to switch.")
    lines.append("Use `/delete <id>` to remove a session.")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def session_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not is_authorized(user.id):
        if update.message:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    if update.message is None:
        return

    user_state = get_user_state(user.id)
    args = context.args or []

    if not args:
        current = user_state.current_session_id or "None"
        await update.message.reply_text(
            f"*Current session:* `{current}`\n\nUse `/session <id>` to switch.",
            parse_mode="Markdown",
        )
        return

    session_id = args[0]
    session = user_state.switch_session(session_id)

    if session:
        await update.message.reply_text(
            f"✅ Switched to: *{session.name}* (`{session.id}`)",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            f"❌ Session `{session_id}` not found. Use `/sessions` to see available.",
            parse_mode="Markdown",
        )


async def delete_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not is_authorized(user.id):
        if update.message:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    if update.message is None:
        return

    user_state = get_user_state(user.id)
    args = context.args or []

    if not args:
        await update.message.reply_text(
            "Usage: `/delete <session_id>`", parse_mode="Markdown"
        )
        return

    session_id = args[0]
    if user_state.delete_session(session_id):
        await update.message.reply_text(
            f"✅ Deleted session `{session_id}`", parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            f"❌ Session `{session_id}` not found.", parse_mode="Markdown"
        )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not is_authorized(user.id):
        if update.message:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    if update.message is None:
        return

    user_state = get_user_state(user.id)

    if (
        user_state.current_session_id
        and user_state.current_session_id in user_state.sessions
    ):
        session = user_state.sessions[user_state.current_session_id]
        session.clear()
        await update.message.reply_text(
            f"✅ Cleared history for *{session.name}*", parse_mode="Markdown"
        )
    else:
        await update.message.reply_text("No active session to clear.")


# =============================================================================
# Account Creation Command
# =============================================================================


async def newaccount_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /newaccount command — create a GitLab trial + Duo OAuth credential."""
    user = update.effective_user
    if not user or not is_authorized(user.id):
        if update.message:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    if update.message is None:
        return

    status_msg = await update.message.reply_text(
        "🔧 *Creating new GitLab Duo account...*\n\n⏳ Initializing automation...",
        parse_mode="Markdown",
    )

    async def update_status(text: str) -> None:
        """Edit the status message with progress updates."""
        try:
            await status_msg.edit_text(text, parse_mode="Markdown")
        except Exception as e:
            logger.warning(f"Failed to update Telegram status message: {e}")

    try:
        # Late imports — these are heavy and optional dependencies
        from rotator_library.providers.gitlab_duo_provider import (
            DEFAULT_OAUTH_CALLBACK_PORT,
            DEFAULT_OAUTH_CLIENT_ID,
            GitLabDuoProvider,
            _get_instance_url,
        )
        from rotator_library.providers.utilities.gitlab_trial_automation import (
            GitLabTrialAutomator,
        )
    except ImportError as e:
        await update_status(
            f"❌ *Import error*\n\n"
            f"Missing dependency: `{e}`\n\n"
            f"Ensure patchright is installed:\n"
            f"`pip install patchright && patchright install chromium`"
        )
        return

    try:
        instance_url = _get_instance_url()
        client_id = os.getenv("GITLAB_OAUTH_CLIENT_ID", DEFAULT_OAUTH_CLIENT_ID)
        callback_port = int(
            os.getenv("GITLAB_DUO_OAUTH_PORT", str(DEFAULT_OAUTH_CALLBACK_PORT))
        )

        if not client_id:
            await update_status(
                "❌ *Configuration error*\n\n"
                "`GITLAB_OAUTH_CLIENT_ID` is not set.\n"
                "Configure it in `.env` and restart the bot."
            )
            return

        # Determine output path
        oauth_dir = Path("oauth_creds")
        oauth_dir.mkdir(exist_ok=True)
        idx = 1
        while (oauth_dir / f"gitlab_duo_oauth_{idx}.json").exists():
            idx += 1
        output_path = str(oauth_dir / f"gitlab_duo_oauth_{idx}.json")

        await update_status(
            "🔧 *Creating new GitLab Duo account...*\n\n"
            "⏳ Setting up temp email and identity..."
        )

        # Build a Console that silently discards Rich output instead of
        # printing to a terminal.  The automator only uses console.print()
        # for status messages — we relay progress via Telegram instead.
        from io import StringIO
        from rich.console import Console as RichConsole

        _quiet_console = RichConsole(file=StringIO(), quiet=True)
        automator = GitLabTrialAutomator(console=_quiet_console)

        async def oauth_runner(auth_url_handler):
            """Run the GitLab OAuth PKCE flow."""
            return await GitLabDuoProvider.oauth_setup(
                instance_url=instance_url,
                client_id=client_id,
                callback_port=callback_port,
                output_path=output_path,
                auth_url_handler=auth_url_handler,
                auto_open_browser=False,
            )

        await update_status(
            "🔧 *Creating new GitLab Duo account...*\n\n"
            "🌐 Launching browser and registering account...\n"
            "This may take a few minutes."
        )

        # Run the full automation
        auto_result = await automator.run(oauth_runner)
        saved_path = auto_result.oauth_path

        # Add proxy metadata to the credential file
        try:
            with open(saved_path, "r") as f:
                creds = json.load(f)

            metadata = creds.setdefault("_proxy_metadata", {})
            metadata["email"] = auto_result.email
            metadata["gitlab_duo_group"] = auto_result.group_path or ""
            metadata["gitlab_trial_automated"] = True
            metadata["created_via"] = "telegram_bot"

            with open(saved_path, "w") as f:
                json.dump(creds, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write proxy metadata to {saved_path}: {e}")

        await update_status(
            "🔧 *Creating new GitLab Duo account...*\n\n"
            f"✅ Account created: `{auto_result.email}`\n"
            f"✅ Credentials saved: `{Path(saved_path).name}`\n\n"
            "⏳ Reloading proxy credentials..."
        )

        # Hot-reload credentials in the proxy
        reload_result = None
        try:
            base_url = get_proxy_base_url()
            url = f"{base_url}/v1/reload-credentials"
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                response = await http_client.post(url, headers=get_auth_headers())
                if response.status_code == 200:
                    reload_result = response.json()
                else:
                    logger.error(
                        f"Credential reload returned HTTP {response.status_code}: "
                        f"{response.text[:200]}"
                    )
        except Exception as e:
            logger.error(f"Failed to call /v1/reload-credentials: {e}")

        # Final success message
        reload_info = ""
        if reload_result and reload_result.get("added"):
            reload_info = (
                f"🔄 Proxy reloaded: +{len(reload_result['added'])} credential(s)\n"
            )
        elif reload_result is not None:
            reload_info = "🔄 Proxy reloaded (credential was already known)\n"
        else:
            reload_info = "⚠️ Could not reload proxy — restart may be needed\n"

        duo_info = ""
        if auto_result.duo_enabled:
            duo_info = f"🤖 Duo enabled for group: `{auto_result.group_path}`\n"

        await update_status(
            f"✅ *New GitLab Duo account ready\\!*\n\n"
            f"📧 Email: `{auto_result.email}`\n"
            f"👤 Username: `{auto_result.username}`\n"
            f"📁 Credentials: `{Path(saved_path).name}`\n"
            f"{duo_info}"
            f"{reload_info}\n"
            f"The new credential is live and accepting requests\\."
        )

    except Exception as e:
        logger.exception("newaccount_command failed")
        error_msg = str(e)
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        # Escape MarkdownV2 special characters in the error message
        escaped_error = error_msg
        for char in "_*[]()~`>#+-=|{}.!":
            escaped_error = escaped_error.replace(char, f"\\{char}")
        await update_status(
            f"❌ *Account creation failed*\n\n```\n{escaped_error}\n```"
        )


# =============================================================================
# Chat Message Handler
# =============================================================================


async def stream_llm_response(
    model: str,
    messages: List[Dict[str, str]],
    update: Update,
    response_msg: Any,
) -> Optional[str]:
    base_url = get_proxy_base_url()
    url = f"{base_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    headers = get_auth_headers()
    headers["Content-Type"] = "application/json"

    full_response = ""
    last_edit_time = 0.0
    edit_interval = 1.0

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0)
        ) as client:
            async with client.stream(
                "POST", url, headers=headers, json=payload
            ) as response:
                if response.status_code == 401:
                    await response_msg.edit_text("❌ Authentication failed.")
                    return None
                elif response.status_code != 200:
                    error_text = await response.aread()
                    await response_msg.edit_text(
                        f"❌ HTTP {response.status_code}: {error_text[:200].decode()}"
                    )
                    return None

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_response += content
                                now = time.time()
                                if now - last_edit_time >= edit_interval:
                                    display = (
                                        full_response[:4000] + "..."
                                        if len(full_response) > 4000
                                        else full_response
                                    )
                                    try:
                                        await response_msg.edit_text(display or "...")
                                    except Exception:
                                        pass
                                    last_edit_time = now
                        except json.JSONDecodeError:
                            continue

        return full_response

    except httpx.ConnectError:
        await response_msg.edit_text("❌ Connection failed. Is the proxy running?")
        return None
    except httpx.TimeoutException:
        await response_msg.edit_text("❌ Request timed out.")
        return None
    except Exception as e:
        logger.exception("Streaming error")
        await response_msg.edit_text(f"❌ Error: {str(e)[:200]}")
        return None


async def chat_message_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    user = update.effective_user
    if not user or not is_authorized(user.id):
        return

    if update.message is None:
        return

    user_text = update.message.text
    if not user_text:
        return

    user_state = get_user_state(user.id)

    if not user_state.selected_model:
        await update.message.reply_text(
            "⚠️ No model selected. Use `/models` to see available models and `/model <name>` to select one.",
            parse_mode="Markdown",
        )
        return

    session = user_state.get_or_create_session()
    session.add_message("user", user_text)

    system_prompt = load_system_prompt()
    api_messages = [{"role": "system", "content": system_prompt}]
    api_messages.extend(session.get_messages_for_api())

    response_msg = await update.message.reply_text("⏳ Thinking...")

    full_response = await stream_llm_response(
        model=user_state.selected_model,
        messages=api_messages,
        update=update,
        response_msg=response_msg,
    )

    if full_response:
        session.add_message("assistant", full_response)

        if len(full_response) > 4000:
            chunks = chunk_message(full_response)
            await response_msg.edit_text(chunks[0])
            for chunk in chunks[1:]:
                await update.message.reply_text(chunk)
        else:
            await response_msg.edit_text(full_response)


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

    # Callback handlers for inline buttons
    application.add_handler(
        CallbackQueryHandler(refresh_callback, pattern=r"^refresh:")
    )
    application.add_handler(CallbackQueryHandler(view_callback, pattern=r"^view:"))

    # LLM Chat handlers
    application.add_handler(CommandHandler("models", models_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("new", new_command))
    application.add_handler(CommandHandler("sessions", sessions_command))
    application.add_handler(CommandHandler("session", session_command))
    application.add_handler(CommandHandler("delete", delete_command))
    application.add_handler(CommandHandler("clear", clear_command))

    # Account management handlers
    application.add_handler(CommandHandler("newaccount", newaccount_command))

    # Message handler for chat (must be last - catches all text messages)
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, chat_message_handler)
    )

    # Run the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
