import asyncio
import email
import imaplib
import json
import logging
import os
import random
import re
import smtplib
import ssl
import time
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from email.message import EmailMessage
from email.parser import BytesParser
from email.policy import default as email_default_policy
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Set
from uuid import uuid4

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import discord
from discord import app_commands

VERSION = "1.0"
REPO_SLUG = "Ikelene/discord-shapes-ai"
REPO_URL = f"https://github.com/{REPO_SLUG}"
LATEST_API = f"https://api.github.com/repos/{REPO_SLUG}/releases/latest"

LOG_LEVEL = logging.INFO if os.getenv("DEBUG_LOG", "1") in ("0", "false", "False") else logging.DEBUG
logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s")
log = logging.getLogger("shape-bot")
logging.getLogger("discord").setLevel(logging.WARNING)
logging.getLogger("discord.http").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)

def need(var: str) -> str:
    val = os.getenv(var)
    if not val:
        log.error("Missing required environment variable: %s", var)
        raise SystemExit(1)
    return val

DISCORD_TOKEN = need("DISCORD_TOKEN")
BOT_NAME = os.getenv("BOT_NAME", "Jarvis")
TRIGGER_CHANCE = int(os.getenv("TRIGGER_CHANCE", "5"))
REACT_CHANCE = int(os.getenv("REACT_CHANCE", "10"))
DEBUG_CHAT = os.getenv("DEBUG_CHAT", "1") not in ("0", "false", "False")

SMTP_HOST = os.getenv("SMTP_HOST", "mail.cadex.dev")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = need("SMTP_USER")
SMTP_PASS = need("SMTP_PASS")
SMTP_TIMEOUT = int(os.getenv("SMTP_TIMEOUT", "45"))
SMTP_HOST_FALLBACKS = [h.strip() for h in os.getenv("SMTP_HOST_FALLBACKS", "").split(",") if h.strip()]

IMAP_HOST = os.getenv("IMAP_HOST", "mail.cadex.dev")
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))
IMAP_USER = os.getenv("IMAP_USER", SMTP_USER)
IMAP_PASS = os.getenv("IMAP_PASS", SMTP_PASS)
IMAP_TIMEOUT = int(os.getenv("IMAP_TIMEOUT", "45"))
IMAP_HOST_FALLBACKS = [h.strip() for h in os.getenv("IMAP_HOST_FALLBACKS", "").split(",") if h.strip()]

IMAP_CANDIDATES = [m.strip() for m in os.getenv(
    "IMAP_MAILBOXES",
    'INBOX,Inbox,inbox,INBOX.Junk,INBOX.Spam,Junk,Spam,"Junk E-mail","Bulk Mail","All Mail",Archive'
).split(',') if m.strip()]

TO_EMAIL = os.getenv("TO_EMAIL", "jarvis-gooner@shapes.inc")
FROM_NAME = os.getenv("FROM_NAME", f"{BOT_NAME} Bridge")
REPLY_WAIT_SECONDS = int(os.getenv("REPLY_WAIT_SECONDS", "60"))
POLL_INTERVAL_SEC = max(1, int(os.getenv("POLL_INTERVAL_SEC", "5")))
APPEND_TO_SENT = os.getenv("APPEND_TO_SENT", "0") not in ("0", "false", "False")
SENT_MAILBOX = os.getenv("SENT_MAILBOX", "Sent")

DATA_DIR = Path(__file__).resolve().parent
GOON_FILE = DATA_DIR / "goon_channels.json"
DL_DIR = DATA_DIR / "_dl"
DL_DIR.mkdir(exist_ok=True)

CLEARY_EMOJI_STR = "<:clearly:1404581300517736558>"
REACTION_CHOICES: List[str] = ["🪑", "🍆", "💦", "😩", "🤑", CLEARY_EMOJI_STR]

class SafeJSON:
    def __init__(self, path: Path, default):
        self.path = path
        self.default = default
        self._lock = asyncio.Lock()
    async def load(self):
        async with self._lock:
            if not self.path.exists():
                return self.default() if callable(self.default) else self.default
            try:
                return json.loads(self.path.read_text())
            except Exception:
                return self.default() if callable(self.default) else self.default
    async def save(self, obj):
        async with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(obj, indent=2))
            tmp.replace(self.path)

goon_store = SafeJSON(GOON_FILE, lambda: [])

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

class BotClient(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
    async def setup_hook(self) -> None:
        await self._sync_commands()
    async def on_ready(self):
        log.info("Logged in as %s (%s)", self.user, self.user.id)
        await self._sync_commands()
    async def on_guild_available(self, guild: discord.Guild):
        await self._sync_commands()
    async def on_guild_join(self, guild: discord.Guild):
        await self._sync_commands()
    async def _sync_commands(self):
        try:
            await self.tree.sync()
            log.info("Slash commands synced.")
        except Exception:
            log.exception("Slash command sync failed")

client = BotClient()

def is_in_goon_channels(guild_id: int, channel_id: int, pairs: List[Dict]) -> bool:
    return any(p["guild_id"] == guild_id and p["channel_id"] == channel_id for p in pairs)

async def add_goon_channel(guild_id: int, channel_id: int):
    data = await goon_store.load()
    if not is_in_goon_channels(guild_id, channel_id, data):
        data.append({"guild_id": guild_id, "channel_id": channel_id})
        await goon_store.save(data)

async def remove_goon_channel(guild_id: int, channel_id: int):
    data = await goon_store.load()
    data = [p for p in data if not (p["guild_id"] == guild_id and p["channel_id"] == channel_id)]
    await goon_store.save(data)

def render_discord_text(message: discord.Message) -> str:
    content = message.content or ""
    for m in message.mentions:
        for pat in (f"<@{m.id}>", f"<@!{m.id}>"):
            content = content.replace(pat, f"@!{m.display_name}!")
    for r in message.role_mentions:
        content = content.replace(f"<@&{r.id}>", f"@{r.name}")
    if hasattr(message, "channel_mentions"):
        for ch in message.channel_mentions:
            content = content.replace(f"<#{ch.id}>", f"#{ch.name}")
    def _chan_sub(m: re.Match) -> str:
        cid = int(m.group(1))
        ch = message.guild.get_channel(cid) if message.guild else None
        return f"#{ch.name}" if ch else f"#unknown-{cid}"
    content = re.sub(r"<#(\d+)>", _chan_sub, content)
    return content.replace(":clearly:", CLEARY_EMOJI_STR)

def strip_quoted_reply(text: str) -> str:
    lines = text.splitlines()
    out: List[str] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if i == 0 and s.startswith("Sent from my "):
            continue
        if s.startswith("On ") and s.endswith("wrote:"):
            break
        if s.startswith("-----Original Message-----"):
            break
        if s.startswith(">"):
            continue
        if s.lower().startswith("from: ") and i > 0:
            break
        out.append(line)
    cleaned = "\n".join(out).strip()
    if not cleaned:
        for line in lines:
            if not line.strip().startswith(">"):
                cleaned = line.strip()
                if cleaned:
                    break
    return cleaned

def _host_candidates(host: str, extra: List[str]) -> List[str]:
    cands: List[str] = []
    if host:
        cands.append(host)
        if not host.lower().startswith("mail."):
            cands.append(f"mail.{host}")
    for h in extra:
        if h and h not in cands:
            cands.append(h)
    return cands

def _smtp_send(em: EmailMessage) -> None:
    ctx = ssl.create_default_context()
    tried: List[str] = []
    for host in _host_candidates(SMTP_HOST, SMTP_HOST_FALLBACKS):
        try:
            tried.append(f"{host}:{SMTP_PORT}")
            log.debug("SMTP connect host=%s port=%s", host, SMTP_PORT)
            if SMTP_PORT == 465:
                with smtplib.SMTP_SSL(host, SMTP_PORT, timeout=SMTP_TIMEOUT, context=ctx) as s:
                    s.ehlo(); s.login(SMTP_USER, SMTP_PASS); s.send_message(em)
            else:
                with smtplib.SMTP(host, SMTP_PORT, timeout=SMTP_TIMEOUT) as s:
                    s.ehlo(); s.starttls(context=ctx); s.ehlo(); s.login(SMTP_USER, SMTP_PASS); s.send_message(em)
            log.debug("SMTP sent ok via %s", host)
            return
        except Exception as e:
            log.debug("SMTP failed via %s: %s", host, e)
            continue
    raise RuntimeError(f"SMTP send failed; tried: {', '.join(tried)}")

def _imap_quote(mailbox: str) -> str:
    return '"' + mailbox.replace('\\', r'\\').replace('"', r'\"') + '"'

def _imap_connect() -> imaplib.IMAP4_SSL:
    ctx = ssl.create_default_context()
    last_err: Optional[Exception] = None
    for host in _host_candidates(IMAP_HOST, IMAP_HOST_FALLBACKS):
        try:
            log.debug("IMAP connecting host=%s port=%s", host, IMAP_PORT)
            M = imaplib.IMAP4_SSL(host, IMAP_PORT, ssl_context=ctx, timeout=IMAP_TIMEOUT)
            M.login(IMAP_USER, IMAP_PASS)
            log.debug("IMAP login ok host=%s", host)
            return M
        except Exception as e:
            last_err = e
            log.debug("IMAP failed host=%s err=%s", host, e)
            continue
    raise RuntimeError(f"IMAP connect/login failed: {last_err}")

def _mailboxes_to_scan() -> List[str]:
    base = ["INBOX","Inbox","inbox","INBOX.Junk","INBOX.Spam","Junk","Spam","Junk E-mail","Bulk Mail","All Mail","Archive"]
    seen: Set[str] = set()
    out: List[str] = []
    for n in ["INBOX"] + IMAP_CANDIDATES + base:
        if n and n not in seen:
            seen.add(n); out.append(n)
    log.debug("IMAP scan order: %s", ", ".join(out))
    return out

def _search_terms(M: imaplib.IMAP4_SSL, *terms: str) -> List[bytes]:
    def _q(x: str) -> str:
        return '"' + x.replace('"','\\"') + '"'
    args = []
    for t in terms:
        if t in {'SINCE','FROM','SUBJECT','HEADER','TEXT','BODY','OR','NOT'}:
            args.append(t)
        else:
            args.append(_q(t))
    for charset in (None, 'UTF-8'):
        try:
            typ, data = M.search(None if charset is None else charset, *args)
            if typ == 'OK':
                ids = (data[0] or b'').split()
                log.debug("IMAP SEARCH %s -> %d hits", " ".join(args), len(ids))
                return ids
        except Exception as e:
            log.debug("IMAP SEARCH error charset=%s: %s", charset, e)
            continue
    return []

def _imap_date_str(dt: datetime) -> str:
    d = dt.astimezone(timezone.utc)
    return d.strftime("%d-%b-%Y")

def _fetch_msgs(M: imaplib.IMAP4_SSL, seq_ids: List[bytes], limit: int = 80) -> List[Tuple[bytes, EmailMessage]]:
    if not seq_ids:
        return []
    try:
        seq_sorted = sorted(seq_ids, key=lambda b: int(b.decode()), reverse=True)
    except Exception:
        seq_sorted = list(reversed(seq_ids))
    out: List[Tuple[bytes, EmailMessage]] = []
    for mid in seq_sorted[:limit]:
        t3, msg_data = M.fetch(mid, '(RFC822)')
        if t3 != 'OK' or not msg_data or not isinstance(msg_data[0], tuple):
            continue
        raw = msg_data[0][1]
        try:
            msg = BytesParser(policy=email_default_policy).parsebytes(raw)
            out.append((mid, msg))
        except Exception as e:
            log.debug("IMAP parse error mid=%s: %s", mid, e)
            continue
    log.debug("IMAP fetched %d messages", len(out))
    return out

def _msg_matches(msg: EmailMessage, token: str, since_dt: datetime, msgid: Optional[str]) -> bool:
    try:
        mdate = msg.get('Date')
        if mdate:
            d = email.utils.parsedate_to_datetime(mdate)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            if d < since_dt:
                return False
    except Exception:
        pass
    subj = str(msg.get('Subject',''))
    if token and token in subj:
        return True
    if msgid:
        msgid_nobr = msgid.strip('<>')
        for hdr in ('In-Reply-To','References','X-Bridge-Message-ID'):
            v = str(msg.get(hdr,''))
            if msgid_nobr and msgid_nobr in v:
                return True
    try:
        body = ''
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type(); disp = (part.get('Content-Disposition') or '').lower()
                if ctype == 'text/plain' and 'attachment' not in disp:
                    body = part.get_content() or ''
                    break
        else:
            if msg.get_content_type() == 'text/plain':
                body = msg.get_content() or ''
        if token and token in body:
            return True
    except Exception:
        pass
    return False

def _force_select(M: imaplib.IMAP4_SSL, mailbox: str) -> bool:
    for candidate in (mailbox, _imap_quote(mailbox)):
        try:
            typ, _ = M.select(candidate)
            log.debug("IMAP select %s -> %s", candidate, typ)
            if typ == 'OK':
                return True
        except Exception as e:
            log.debug("IMAP select error %s: %s", candidate, e)
    return False

def _imap_wait_for_reply(token: str, since_dt: datetime, msgid: Optional[str]=None) -> Optional[EmailMessage]:
    deadline = time.time() + REPLY_WAIT_SECONDS
    since_str = _imap_date_str(since_dt)
    log.debug("WAIT start token=%s since=%s msgid=%s timeout=%ss to_email=%s", token, since_dt.isoformat(), msgid, REPLY_WAIT_SECONDS, TO_EMAIL)
    while time.time() < deadline:
        try:
            with _imap_connect() as M:
                for mailbox in _mailboxes_to_scan():
                    if not _force_select(M, mailbox):
                        continue
                    hit_ids: List[bytes] = []
                    if msgid:
                        mid = msgid.strip('<>')
                        hit_ids += _search_terms(M, 'HEADER','In-Reply-To', mid)
                        hit_ids += _search_terms(M, 'HEADER','References', mid)
                        hit_ids += _search_terms(M, 'HEADER','X-Bridge-Message-ID', mid)
                    hit_ids += _search_terms(M, 'FROM', TO_EMAIL)
                    if token:
                        hit_ids += _search_terms(M, 'SUBJECT', token)
                        hit_ids += _search_terms(M, 'BODY', token)
                        hit_ids += _search_terms(M, 'TEXT', token)
                    seen: Set[bytes] = set(); ordered: List[bytes] = []
                    for midb in reversed(hit_ids):
                        if midb not in seen:
                            seen.add(midb); ordered.append(midb)
                    pairs = _fetch_msgs(M, ordered, limit=120)
                    for midb, msg in pairs:
                        if _msg_matches(msg, token, since_dt, msgid):
                            try:
                                M.store(midb, '+FLAGS', r'(\\Deleted)'); M.expunge()
                            except Exception:
                                pass
                            M.logout()
                            return msg
                    typ, data = M.search(None, 'SINCE', since_str)
                    seqs = (data[0] or b'').split() if typ == 'OK' else []
                    if seqs:
                        pairs = _fetch_msgs(M, seqs, limit=200)
                        for midb, msg in pairs:
                            if _msg_matches(msg, token, since_dt, msgid):
                                try:
                                    M.store(midb, '+FLAGS', r'(\\Deleted)'); M.expunge()
                                except Exception:
                                    pass
                                M.logout()
                                return msg
        except Exception as e:
            log.debug("IMAP poll error: %s", e)
        time.sleep(POLL_INTERVAL_SEC)
    log.debug("WAIT timeout token=%s", token)
    return None

def _build_email(subject: str, body: str, message_id: str) -> EmailMessage:
    em = EmailMessage()
    em['From'] = f"{FROM_NAME} <{SMTP_USER}>"
    em['To'] = TO_EMAIL
    em['Subject'] = subject
    em['Message-ID'] = message_id
    em['X-Bridge-Message-ID'] = message_id
    em.set_content(body)
    return em

def _imap_append_sent(em: EmailMessage) -> None:
    if not APPEND_TO_SENT:
        return
    try:
        with _imap_connect() as M:
            try:
                M.append(SENT_MAILBOX, '("Seen")', imaplib.Time2Internaldate(time.time()), em.as_bytes())
                log.debug("IMAP appended to %s", SENT_MAILBOX)
            finally:
                M.logout()
    except Exception as e:
        log.debug("IMAP append sent error: %s", e)

def _version_tuple(s: str) -> Tuple[int, ...]:
    clean = re.sub(r"[^0-9.]", "", s or "")
    parts = [p for p in clean.split(".") if p != ""]
    return tuple(int(p) for p in parts) if parts else (0,)

def _check_latest_and_pause():
    try:
        req = urllib.request.Request(LATEST_API, headers={"User-Agent": f"{BOT_NAME}/{VERSION}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8", "ignore"))
        latest_title = str(data.get("name") or data.get("tag_name") or "").strip()
        if not latest_title:
            return
        cur = _version_tuple(VERSION)
        latest = _version_tuple(latest_title)
        if latest and cur < latest:
            print(f"A newer version ({latest_title}) is available. Update when you can:\n{REPO_URL}")
            input("Press Enter to continue...")
    except Exception:
        pass

def _is_file_url(u: str) -> bool:
    try:
        p = urllib.parse.urlparse(u.strip())
        if p.scheme not in ("http", "https"):
            return False
        name = os.path.basename(p.path).lower()
        return any(name.endswith(ext) for ext in (".mp3",".wav",".m4a",".flac",".ogg",".mp4",".mov",".mkv",".png",".jpg",".jpeg",".gif",".webp",".pdf",".txt",".zip",".rar",".7z"))
    except Exception:
        return False

def _extract_urls(text: str) -> List[str]:
    if not text:
        return []
    pat = r'(https?://[^\s<>")]+)'
    return re.findall(pat, text)

def _download_to_file(url: str) -> Optional[Path]:
    try:
        parsed = urllib.parse.urlparse(url)
        name = os.path.basename(parsed.path) or f"file-{uuid4().hex}"
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        dest = DL_DIR / safe_name
        req = urllib.request.Request(url, headers={"User-Agent": f"{BOT_NAME}/{VERSION}"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as out:
            out.write(r.read())
        log.debug("Downloaded file %s from %s", dest, url)
        return dest
    except Exception as e:
        log.debug("Download error %s: %s", url, e)
        return None

def _split_text_and_files(reply_text: str) -> Tuple[str, List[Path], List[str]]:
    urls = _extract_urls(reply_text)
    files: List[Path] = []
    kept_urls: List[str] = []
    for u in urls:
        if _is_file_url(u):
            fp = _download_to_file(u)
            if fp and fp.exists():
                files.append(fp)
            else:
                kept_urls.append(u)
        else:
            kept_urls.append(u)
    text_without_urls = reply_text
    for u in urls:
        text_without_urls = text_without_urls.replace(u, "").strip()
    return (text_without_urls.strip(), files, kept_urls)

def _smtp_send_and_wait_reply(body: str) -> Tuple[Optional[Dict], float]:
    t0 = time.monotonic()
    token = uuid4().hex[:12]
    subject = f"{BOT_NAME} Bridge [{token}]"
    message_id = f"<{uuid4().hex}@{BOT_NAME.lower()}-bridge>"
    em = _build_email(subject, body, message_id)
    sent_at = datetime.now(tz=timezone.utc)
    log.debug("SEND start token=%s subject=%s msgid=%s to=%s", token, subject, message_id, TO_EMAIL)
    try:
        _smtp_send(em)
        _imap_append_sent(em)
    except Exception as e:
        log.debug("SMTP fatal: %s", e)
        return None, time.monotonic() - t0
    time.sleep(1)
    reply_msg: Optional[EmailMessage] = _imap_wait_for_reply(token, sent_at, message_id)
    dur = time.monotonic() - t0
    if not reply_msg:
        return {'ok': True, 'uuid': token, 'status': 'timeout', 'reply': f"⚠️ {BOT_NAME} didn't reply in time (60s). Try again shortly."}, dur
    def _extract_text(msg: EmailMessage) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type(); disp = (part.get('Content-Disposition') or '').lower()
                if ctype == 'text/plain' and 'attachment' not in disp:
                    try: return part.get_content()
                    except Exception: pass
            for part in msg.walk():
                if part.get_content_type() == 'text/html':
                    try:
                        html = part.get_content()
                        return re.sub(r'<[^>]+>', '', html)
                    except Exception: pass
        else:
            if msg.get_content_type() == 'text/plain':
                try: return msg.get_content()
                except Exception: pass
            if msg.get_content_type() == 'text/html':
                try:
                    html = msg.get_content()
                    return re.sub(r'<[^>]+>', '', html)
                except Exception: pass
        return ''
    text = strip_quoted_reply((_extract_text(reply_msg) or '').strip()) or '(empty reply)'
    subj = str(reply_msg.get('Subject',''))
    frm = str(reply_msg.get('From',''))
    log.debug("RECV subject=%s from=%s token=%s dur=%.2fs", subj, frm, token, dur)
    return {'ok': True, 'uuid': token, 'status': 'got_reply', 'reply': text, 'from': frm, 'subject': subj, 'receivedAt': str(reply_msg.get('Date',''))}, dur

async def maybe_react(message: discord.Message) -> None:
    try:
        if REACT_CHANCE <= 0:
            return
        if random.randint(1, TRIGGER_CHANCE) != 1:
            return
        choice = random.choice(REACTION_CHOICES)
        if choice.startswith('<:') and choice.endswith('>'):
            try:
                pe = discord.PartialEmoji.from_str(choice)
                await message.add_reaction(pe)
                return
            except Exception:
                pass
        await message.add_reaction(choice)
    except Exception:
        pass

@client.event
async def on_message(message: discord.Message):
    if client.user and message.author.id == client.user.id:
        return
    if message.author.bot:
        return
    if not message.guild:
        display_name = message.author.display_name or message.author.name
        base = f"From {display_name} (DM)\n"
        body = (base + "\n" + (message.content or "")).strip()
        async with message.channel.typing():
            data, dur = await asyncio.to_thread(_smtp_send_and_wait_reply, body)
        if data and data.get('reply'):
            reply_text, files, leftover_urls = _split_text_and_files(data['reply'])
            sent_any = False
            if files:
                try:
                    await message.reply(files=[discord.File(str(fp), filename=fp.name) for fp in files[:10]])
                    sent_any = True
                except Exception:
                    pass
            content_parts = []
            if reply_text:
                content_parts.append(reply_text[:1900])
            if leftover_urls:
                content_parts.append("\n".join(leftover_urls)[:1900])
            content = "\n".join([p for p in content_parts if p]).strip()
            if content:
                try:
                    await message.reply(content=content)
                    sent_any = True
                except Exception:
                    pass
            if not sent_any:
                try:
                    await message.reply(content=data['reply'][:1900])
                except Exception:
                    pass
        return

    goon_pairs = await goon_store.load()
    guild_entries = [p for p in goon_pairs if p["guild_id"] == message.guild.id]
    enabled_here = is_in_goon_channels(message.guild.id, message.channel.id, goon_pairs)

    content_lower = (message.content or '').lower()
    mentioned_me = client.user in message.mentions if client.user else False
    replied_to_bot = False
    if message.reference and message.reference.resolved:
        ref = message.reference.resolved
        if isinstance(ref, discord.Message) and client.user and ref.author.id == client.user.id:
            replied_to_bot = True
    contains_name = BOT_NAME.lower() in content_lower
    random_fire = (random.randint(1, TRIGGER_CHANCE) == 1)

    bootstrap_ok = (not guild_entries) and mentioned_me
    if not (enabled_here or bootstrap_ok):
        return

    await maybe_react(message)

    reason = None
    if mentioned_me:
        reason = 'mention'
    elif replied_to_bot:
        reason = 'reply-to-bot'
    elif contains_name:
        reason = f"contains '{BOT_NAME.lower()}'"
    elif random_fire:
        reason = f'random 1/{TRIGGER_CHANCE}'
    if not reason:
        return

    display_name = message.author.display_name
    base = f"From {display_name} (#{message.channel.name})\n"
    if replied_to_bot and message.reference and message.reference.resolved:
        ref_msg: discord.Message = message.reference.resolved  # type: ignore
        snippet = (ref_msg.content or '').replace('\n',' ')[:200] or '[no text]'
        base += f"Replied to: \"{snippet}\"\n"
    if mentioned_me:
        base += f"({BOT_NAME} was mentioned)\n"

    sanitized = render_discord_text(message)
    body = (base + "\n" + sanitized).strip()
    async with message.channel.typing():
        data, dur = await asyncio.to_thread(_smtp_send_and_wait_reply, body)

    if data and data.get('reply'):
        reply_text, files, leftover_urls = _split_text_and_files(data['reply'])
        sent_any = False
        if files:
            try:
                await message.reply(files=[discord.File(str(fp), filename=fp.name) for fp in files[:10]])
                sent_any = True
            except Exception:
                pass
        content_parts = []
        if reply_text:
            content_parts.append(reply_text[:1900])
        if leftover_urls:
            content_parts.append("\n".join(leftover_urls)[:1900])
        content = "\n".join([p for p in content_parts if p]).strip()
        if content:
            try:
                await message.reply(content=content)
                sent_any = True
            except Exception:
                pass
        if not sent_any:
            try:
                await message.reply(content=data['reply'][:1900])
            except Exception:
                pass

@client.tree.command(name="enable", description=lambda: f"Enable {BOT_NAME} in a channel" if True else "Enable bot")
@app_commands.describe(channel="Channel to enable")
async def enable_cmd(interaction: discord.Interaction, channel: discord.TextChannel):
    await interaction.response.defer(ephemeral=True, thinking=False)
    if not interaction.guild:
        await interaction.edit_original_response(content="This command only works in a server.")
        return
    if not isinstance(interaction.user, discord.Member):
        await interaction.edit_original_response(content="Could not resolve your member permissions.")
        return
    perms = interaction.user.guild_permissions
    if not (perms.administrator or perms.manage_guild):
        await interaction.edit_original_response(content="You need Administrator or Manage Server to use this.")
        return
    await add_goon_channel(interaction.guild_id, channel.id)  # type: ignore
    await interaction.edit_original_response(content=f"Enabled {BOT_NAME} in {channel.mention}.")

@client.tree.command(name="disable", description=lambda: f"Disable {BOT_NAME} in a channel" if True else "Disable bot")
@app_commands.describe(channel="Channel to disable")
async def disable_cmd(interaction: discord.Interaction, channel: discord.TextChannel):
    await interaction.response.defer(ephemeral=True, thinking=False)
    if not interaction.guild:
        await interaction.edit_original_response(content="This command only works in a server.")
        return
    if not isinstance(interaction.user, discord.Member):
        await interaction.edit_original_response(content="Could not resolve your member permissions.")
        return
    perms = interaction.user.guild_permissions
    if not (perms.administrator or perms.manage_guild):
        await interaction.edit_original_response(content="You need Administrator or Manage Server to use this.")
        return
    await remove_goon_channel(interaction.guild_id, channel.id)  # type: ignore
    await interaction.edit_original_response(content=f"Disabled {BOT_NAME} in {channel.mention}.")

@client.tree.command(name="goonstats", description=lambda: f"Show {BOT_NAME} channel stats for this server" if True else "Show stats")
async def goon_stats_cmd(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=False)
    if not interaction.guild:
        await interaction.edit_original_response(content="This command only works in a server.")
        return
    data = await goon_store.load()
    here = [p for p in data if p['guild_id'] == interaction.guild_id]
    if not here:
        await interaction.edit_original_response(content=f"No channels enabled for {BOT_NAME} in this server.")
        return
    mentions = []
    for p in here:
        ch = interaction.guild.get_channel(p['channel_id']) if interaction.guild else None  # type: ignore
        if ch:
            mentions.append(ch.mention)
    await interaction.edit_original_response(content=f"Enabled channels ({len(mentions)}): " + ", ".join(mentions))

for cmd in list(client.tree.get_commands()):
    if callable(cmd.description):
        cmd.description = cmd.description()

async def main():
    try:
        req = urllib.request.Request(LATEST_API, headers={"User-Agent": f"{BOT_NAME}/{VERSION}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8", "ignore"))
        latest_title = str(data.get("name") or data.get("tag_name") or "").strip()
        cur = _version_tuple(VERSION)
        latest = _version_tuple(latest_title)
        if latest and cur < latest:
            print(f"A newer version ({latest_title}) is available. Update when you can:\n{REPO_URL}")
            input("Press Enter to continue...")
    except Exception:
        pass
    await client.start(DISCORD_TOKEN)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
