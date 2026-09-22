"""
Notifier Telegram sederhana — stdlib urllib, tanpa dependency tambahan.

Setup sekali:
1. Chat @BotFather di Telegram -> /newbot -> ikuti langkahnya -> salin TOKEN.
2. Kirim pesan apa saja (mis. "halo") ke bot kamu dari akun Telegram.
3. Buka di browser: https://api.telegram.org/bot<TOKEN>/getUpdates
   -> cari "chat":{"id": 123456789, ...} -> itulah CHAT_ID kamu.

Catatan Bot API:
- sendDocument: batas ukuran file 50 MB.
- Bot hanya bisa mengirim ke user yang pernah mengirim pesan ke bot itu
  (makanya langkah 2 wajib).
- TOKEN = kredensial bot. Jangan dibagikan; kalau bocor, /revoke di BotFather.
"""

import json
import mimetypes
import os
import uuid
from urllib.request import Request, urlopen

API_BASE = "https://api.telegram.org/bot{token}/{method}"


def _post(token, method, payload_bytes, content_type):
    url = API_BASE.format(token=token, method=method)
    req = Request(url, data=payload_bytes, headers={"Content-Type": content_type})
    with urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data}")
    return data


def send_message(token, chat_id, text):
    payload = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    return _post(token, "sendMessage", payload, "application/json")


def send_document(token, chat_id, filepath, caption=None):
    """Kirim file via multipart/form-data (tanpa library eksternal)."""
    boundary = uuid.uuid4().hex
    filename = os.path.basename(filepath)
    mime = mimetypes.guess_type(filepath)[0] or "application/octet-stream"

    with open(filepath, "rb") as f:
        file_bytes = f.read()

    parts = []

    def field(name, value):
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )

    field("chat_id", str(chat_id))
    if caption:
        field("caption", caption)
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="document"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    body = b"".join(parts)
    return _post(token, "sendDocument", body, f"multipart/form-data; boundary={boundary}")


class TelegramNotifier:
    """Wrapper aman: kegagalan Telegram TIDAK BOLEH menghentikan scraping.

    Token/chat_id bisa dari argumen CLI atau environment variable
    TG_BOT_TOKEN / TG_CHAT_ID. Kalau tidak ada keduanya, notifier nonaktif
    dan scraper jalan seperti biasa.
    """

    def __init__(self, token=None, chat_id=None):
        self.token = token or os.environ.get("TG_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TG_CHAT_ID")

    @property
    def enabled(self):
        return bool(self.token and self.chat_id)

    def notify(self, text):
        if not self.enabled:
            return False
        try:
            send_message(self.token, self.chat_id, text)
            return True
        except Exception as e:
            print(f"\n⚠️ Gagal kirim pesan Telegram: {e}")
            return False

    def send_file(self, filepath, caption=None):
        if not self.enabled:
            return False
        try:
            send_document(self.token, self.chat_id, filepath, caption=caption)
            return True
        except Exception as e:
            print(f"\n⚠️ Gagal kirim file Telegram: {e}")
            return False
