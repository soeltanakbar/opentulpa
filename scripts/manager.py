import asyncio
import os
import re
import secrets
import signal
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv():
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")


# Load before anything else
load_dotenv()

LOG_DIR = REPO_ROOT / ".opentulpa" / "logs"
APP_LOG = LOG_DIR / "app.log"
TUNNEL_LOG = LOG_DIR / "cloudflared.log"
STARTUP_WAIT_SECONDS = int(os.environ.get("STARTUP_WAIT_SECONDS", "180"))


class TulpaManager:
    def __init__(self):
        self.app_proc: subprocess.Popen | None = None
        self.tunnel_proc: subprocess.Popen | None = None
        self.stopping = False

    def log(self, msg: str):
        print(f"[manager] {msg}")

    def error(self, msg: str):
        print(f"[error] {msg}", file=sys.stderr)

    def cleanup_stale_processes(self):
        """Kill any processes listening on our ports."""
        self.log("cleaning up stale processes...")
        ports = [8000]
        for port in ports:
            try:
                # Find PIDs listening on port
                result = subprocess.run(
                    ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True
                )
                pids = result.stdout.strip().split()
                for pid in pids:
                    if pid:
                        self.log(f"killing process {pid} on port {port}")
                        subprocess.run(["kill", "-9", pid], check=False)
            except Exception as e:
                self.log(f"could not check port {port}: {e}")

    def rotate_logs(self):
        self.log("rotating logs...")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        for log_path in [APP_LOG, TUNNEL_LOG]:
            if log_path.exists():
                log_path.replace(log_path.with_suffix(".log.old"))

    async def run(self):
        # 1. Setup
        self.cleanup_stale_processes()
        self.rotate_logs()

        # Boost environment for Gemini 3 and flaky connections
        os.environ["OPENAI_MAX_RETRIES"] = "10"
        os.environ["HTTPX_TIMEOUT"] = "120.0"

        # 2. Launch App
        self.log("launching OpenTulpa app...")
        app_env = os.environ.copy()
        if str(app_env.get("TELEGRAM_BOT_TOKEN", "")).strip() and not str(
            app_env.get("TELEGRAM_WEBHOOK_SECRET", "")
        ).strip():
            app_env["TELEGRAM_WEBHOOK_SECRET"] = secrets.token_urlsafe(24)
            self.log("generated ephemeral TELEGRAM_WEBHOOK_SECRET for this run.")
        if not str(app_env.get("HOST", "")).strip():
            app_env["HOST"] = "127.0.0.1"
            self.log("defaulted HOST=127.0.0.1 for local-only app binding.")
        src_dir = str((REPO_ROOT / "src").resolve())
        existing_pythonpath = app_env.get("PYTHONPATH", "")
        app_env["PYTHONPATH"] = (
            f"{src_dir}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else src_dir
        )
        with open(APP_LOG, "w") as f:
            self.app_proc = subprocess.Popen(
                [sys.executable, "-m", "opentulpa"],
                env=app_env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
            )

        # 3. Wait for Healthz
        self.log("waiting for app to be healthy...")
        import httpx

        healthy = False
        for _ in range(STARTUP_WAIT_SECONDS):
            if self.app_proc.poll() is not None:
                self.error("app exited early. check app.log")
                return
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get("http://127.0.0.1:8000/healthz")
                    if r.status_code == 200:
                        healthy = True
                        break
            except Exception:
                pass
            await asyncio.sleep(1)

        if not healthy:
            self.error("app health check timed out.")
            self.stop()
            return

        # 4. Launch Cloudflare Tunnel
        self.log("launching Cloudflare tunnel...")
        with open(TUNNEL_LOG, "w") as f:
            self.tunnel_proc = subprocess.Popen(
                ["cloudflared", "tunnel", "--url", "http://localhost:8000"],
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        # 5. Extract Tunnel URL
        self.log("extracting tunnel URL...")
        tunnel_url = None
        for _ in range(60):
            if self.tunnel_proc.poll() is not None:
                self.error("tunnel exited early. check cloudflared.log")
                break
            if TUNNEL_LOG.exists():
                content = TUNNEL_LOG.read_text()
                match = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", content)
                if match:
                    tunnel_url = match.group(0)
                    break
            await asyncio.sleep(0.5)  # Faster polling

        if not tunnel_url:
            self.error("could not detect tunnel URL.")
            self.stop()
            return

        self.log(f"tunnel live: {tunnel_url}")

        # 6. Wait for Tunnel Reachability (Public Check)
        self.log("checking tunnel reachability...")
        reachable = False
        async with httpx.AsyncClient() as client:
            for _ in range(40):  # 20 seconds total at 0.5s intervals
                try:
                    r = await client.get(f"{tunnel_url}/healthz", timeout=2.0)
                    if r.status_code == 200:
                        reachable = True
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.5)

        if not reachable:
            self.error("tunnel not reachable yet, but will try setting webhook anyway...")

        # 7. Set Webhook
        webhook_url = f"{tunnel_url}/webhook/telegram"
        bot_token = app_env.get("TELEGRAM_BOT_TOKEN")
        secret = app_env.get("TELEGRAM_WEBHOOK_SECRET")

        if bot_token:
            self.log(f"setting telegram webhook to {webhook_url}...")
            async with httpx.AsyncClient() as client:
                data = {"url": webhook_url}
                if secret:
                    data["secret_token"] = secret
                r = await client.post(
                    f"https://api.telegram.org/bot{bot_token}/setWebhook", data=data
                )
                if r.json().get("ok"):
                    self.log("webhook set successfully.")
                else:
                    self.error(f"failed to set webhook: {r.text}")
        else:
            self.error("TELEGRAM_BOT_TOKEN not found in environment or .env. Webhook skipped.")

        self.log("--- OpenTulpa is live ---")
        self.log(f"Tunnel URL: {tunnel_url}")
        self.log("Press Ctrl+C to shutdown.")

        # 7. Monitor
        while not self.stopping:
            if self.app_proc.poll() is not None:
                self.error("app process died.")
                break
            if self.tunnel_proc.poll() is not None:
                self.error("tunnel process died.")
                break
            await asyncio.sleep(5)

        self.stop()

    def stop(self):
        if self.stopping:
            return
        self.stopping = True
        self.log("shutting down processes...")
        for proc in [self.tunnel_proc, self.app_proc]:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    manager = TulpaManager()

    # Signal handling
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: manager.stop())

    try:
        loop.run_until_complete(manager.run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[fatal] {e}")
    finally:
        manager.stop()
        loop.close()
