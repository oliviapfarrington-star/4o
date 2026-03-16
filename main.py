"""
Discord bot with long-term conversation memory.

Uses:
  - discord.py for the bot framework
  - SQLAlchemy + asyncpg for async Postgres storage
  - OpenAI API (gpt-4o-2024-11-20) for chat completions

Required environment variables (use a .env file or export them):
  DISCORD_BOT_TOKEN   – your Discord bot token
  OPENAI_API_KEY      – your OpenAI API key
  DATABASE_URL        – Postgres connection string
                        e.g. postgresql+asyncpg://user:pass@localhost:5432/botdb
  SYSTEM_PROMPT       – (optional) custom system prompt for the assistant
"""

from __future__ import annotations

import os
import datetime as dt
from typing import Optional

import discord
from discord.ext import commands
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sqlalchemy import (
    BigInteger,
    DateTime,
    Integer,
    String,
    Text,
    select,
    delete,
    func,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

DISCORD_TOKEN: str = os.environ["DISCORD_BOT_TOKEN"]
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
DATABASE_URL: str = os.environ["DATABASE_URL"]

SYSTEM_PROMPT: str = os.getenv(
    "SYSTEM_PROMPT",
    (
        "You are a helpful and friendly assistant in a Discord server. "
        "You remember previous messages from the user across sessions because "
        "the conversation history is stored for you. Be concise, warm, and helpful."
    ),
)

# OpenAI generation parameters
MODEL: str = "gpt-4o-2024-11-20"
TEMPERATURE: float = 0.90
TOP_P: float = 0.92
PRESENCE_PENALTY: float = 0.65
FREQUENCY_PENALTY: float = 0.35
MAX_OUTPUT_TOKENS: int = 1100

# How many past messages to include in the context window per user.
# Adjust to control token usage. Each row is one user or assistant message.
CONTEXT_MESSAGE_LIMIT: int = 80

# ---------------------------------------------------------------------------
# Database models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class Message(Base):
    """Stores every user and assistant message, keyed by Discord user + channel."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    channel_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

engine = create_async_engine(DATABASE_URL, echo=False, pool_size=10, max_overflow=20)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Create tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def save_message(
    user_id: int,
    channel_id: int,
    role: str,
    content: str,
) -> None:
    """Persist a single message to the database."""
    async with async_session() as session:
        session.add(
            Message(
                user_id=user_id,
                channel_id=channel_id,
                role=role,
                content=content,
            )
        )
        await session.commit()


async def fetch_history(
    user_id: int,
    channel_id: int,
    limit: int = CONTEXT_MESSAGE_LIMIT,
) -> list[dict[str, str]]:
    """Return the most recent *limit* messages for a user+channel pair,
    oldest-first, formatted as OpenAI message dicts."""
    async with async_session() as session:
        stmt = (
            select(Message.role, Message.content)
            .where(Message.user_id == user_id, Message.channel_id == channel_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        rows = (await session.execute(stmt)).all()

    # Reverse so oldest comes first
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]


async def clear_history(user_id: int, channel_id: int) -> int:
    """Delete all stored messages for a user+channel. Returns count deleted."""
    async with async_session() as session:
        stmt = delete(Message).where(
            Message.user_id == user_id,
            Message.channel_id == channel_id,
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def get_completion(conversation: list[dict[str, str]]) -> str:
    """Send a conversation to the OpenAI Chat API and return the reply."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *conversation]

    response = await openai_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        presence_penalty=PRESENCE_PENALTY,
        frequency_penalty=FREQUENCY_PENALTY,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Discord bot
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready() -> None:
    await init_db()
    print(f"Logged in as {bot.user}  (ID: {bot.user.id})")  # type: ignore[union-attr]
    print(f"Model: {MODEL} | Temp: {TEMPERATURE} | Top-P: {TOP_P}")
    print(f"Presence Penalty: {PRESENCE_PENALTY} | Frequency Penalty: {FREQUENCY_PENALTY}")
    print(f"Max Output Tokens: {MAX_OUTPUT_TOKENS}")
    print("Database tables ready. Bot is online.")


@bot.event
async def on_message(message: discord.Message) -> None:
    # Ignore messages from bots (including self)
    if message.author.bot:
        return

    # Let commands run first (!clear, !history, etc.)
    await bot.process_commands(message)

    # Skip if this message was a bot command (already handled above)
    ctx: Optional[commands.Context] = await bot.get_context(message)
    if ctx and ctx.valid:
        return

    # Strip any bot mention from the message text
    user_text = message.content
    if bot.user:
        user_text = user_text.replace(f"<@{bot.user.id}>", "").strip()
        user_text = user_text.replace(f"<@!{bot.user.id}>", "").strip()

    if not user_text:
        return

    user_id = message.author.id
    channel_id = message.channel.id

    # Save the user's message
    await save_message(user_id, channel_id, "user", user_text)

    # Build conversation from history
    conversation = await fetch_history(user_id, channel_id)

    # Get the AI reply
    async with message.channel.typing():
        reply = await get_completion(conversation)

    # Save the assistant's reply
    await save_message(user_id, channel_id, "assistant", reply)

    # Discord has a 2 000-char limit per message; split if necessary
    for chunk in split_message(reply):
        await message.reply(chunk, mention_author=False)


def split_message(text: str, limit: int = 2000) -> list[str]:
    """Split text into chunks that fit within Discord's message limit."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at the last newline before the limit
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


# ---------------------------------------------------------------------------
# Bot commands
# ---------------------------------------------------------------------------


@bot.command(name="clear")
async def clear_cmd(ctx: commands.Context) -> None:
    """Wipe your conversation history with the bot in this channel."""
    deleted = await clear_history(ctx.author.id, ctx.channel.id)
    await ctx.reply(
        f"Done — cleared **{deleted}** messages from your history in this channel.",
        mention_author=False,
    )


@bot.command(name="history")
async def history_cmd(ctx: commands.Context, count: int = 10) -> None:
    """Show your last N stored messages (default 10)."""
    rows = await fetch_history(ctx.author.id, ctx.channel.id, limit=count)
    if not rows:
        await ctx.reply("No conversation history found.", mention_author=False)
        return

    lines: list[str] = []
    for msg in rows[-count:]:
        prefix = "🧑" if msg["role"] == "user" else "🤖"
        # Truncate long messages in the preview
        preview = msg["content"][:120] + ("…" if len(msg["content"]) > 120 else "")
        lines.append(f"{prefix} **{msg['role']}**: {preview}")

    text = "\n".join(lines)
    for chunk in split_message(text):
        await ctx.reply(chunk, mention_author=False)


@bot.command(name="info")
async def info_cmd(ctx: commands.Context) -> None:
    """Show current model configuration."""
    embed = discord.Embed(title="Bot Configuration", color=0x7289DA)
    embed.add_field(name="Model", value=MODEL, inline=True)
    embed.add_field(name="Temperature", value=str(TEMPERATURE), inline=True)
    embed.add_field(name="Top-P", value=str(TOP_P), inline=True)
    embed.add_field(name="Presence Penalty", value=str(PRESENCE_PENALTY), inline=True)
    embed.add_field(name="Frequency Penalty", value=str(FREQUENCY_PENALTY), inline=True)
    embed.add_field(name="Max Output Tokens", value=str(MAX_OUTPUT_TOKENS), inline=True)
    embed.add_field(name="Context Window", value=f"{CONTEXT_MESSAGE_LIMIT} messages", inline=True)
    await ctx.reply(embed=embed, mention_author=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
