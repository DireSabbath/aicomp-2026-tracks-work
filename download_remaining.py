# -*- coding: utf-8 -*-
"""Resume-download remaining AIC attachments until complete."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FILES = DATA / "attachments" / "_files"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AIC2026Crawler/1.4"
URLS_PATH = DATA / "asset_urls.json"


def log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()


def fname_of(url: str) -> str:
    name = unquote(os.path.basename(urlparse(url).path) or "file")
    return re.sub(r'[\\/:*?"<>|]', "_", name) or "file"


def curl(args: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["curl.exe", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )


def head_info(url: str) -> tuple[int | None, int | None]:
    try:
        proc = curl(["-sI", "-L", "-g", "--max-time", "25", "-A", UA, url], timeout=35)
    except Exception as e:
        log(f"  HEAD fail {fname_of(url)} ({e})")
        return None, None
    length = None
    status = None
    for line in (proc.stdout or "").splitlines():
        low = line.lower()
        if low.startswith("http/"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                status = int(parts[1])
        if low.startswith("content-length:"):
            try:
                length = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return status, length


def looks_complete(path: Path, expected: int | None) -> bool:
    if not path.exists():
        return False
    size = path.stat().st_size
    if expected and expected > 0:
        return size >= expected
    if size < 4096:
        return False
    head = path.read_bytes()[:8]
    name = path.name.lower()
    if name.endswith(".pdf"):
        return head.startswith(b"%PDF")
    if name.endswith(".docx"):
        return head.startswith(b"PK")
    return size > 8192


def download(url: str, dest: Path, expected: int | None) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    resume = dest.exists() and dest.stat().st_size > 0
    for attempt in range(1, 7):
        if looks_complete(dest, expected):
            return {
                "url": url,
                "file": str(dest),
                "bytes": dest.stat().st_size,
                "expected": expected,
                "ok": True,
                "cached": attempt == 1,
            }
        args = [
            "-sS",
            "-L",
            "-g",
            "--retry",
            "2",
            "--retry-delay",
            "2",
            "--max-time",
            "300",
            "-A",
            UA,
            "-o",
            str(dest),
            url,
        ]
        if resume and dest.exists() and dest.stat().st_size > 0:
            args[2:2] = ["-C", "-"]
        try:
            proc = curl(args, timeout=330)
            size = dest.stat().st_size if dest.exists() else 0
            err = (proc.stderr or "").strip()
            if proc.returncode == 0 and looks_complete(dest, expected):
                return {
                    "url": url,
                    "file": str(dest),
                    "bytes": size,
                    "expected": expected,
                    "ok": True,
                    "cached": False,
                }
            blob = dest.read_bytes()[:32].lower() if dest.exists() else b""
            if blob.startswith(b"<!doctype") or blob.startswith(b"<html"):
                dest.unlink()
                resume = False
                last_err = "got HTML instead of file"
                break
            if proc.returncode == 33 or "does not seem to support byte ranges" in err:
                if looks_complete(dest, expected):
                    return {
                        "url": url,
                        "file": str(dest),
                        "bytes": size,
                        "expected": expected,
                        "ok": True,
                        "cached": False,
                    }
                if dest.exists():
                    dest.unlink()
                resume = False
                last_err = err or "no byte ranges, restart"
                log(f"  restart {fname_of(url)}")
                continue
            last_err = err or f"curl {proc.returncode} size={size} expected={expected}"
        except Exception as e:
            last_err = str(e)
        log(f"  retry {attempt} {fname_of(url)} ({last_err})")
        time.sleep(1.2 * attempt)
    size = dest.stat().st_size if dest.exists() else 0
    return {
        "url": url,
        "file": str(dest) if dest.exists() else None,
        "bytes": size,
        "expected": expected,
        "ok": looks_complete(dest, expected),
        "error": last_err,
    }


def main() -> None:
    urls = json.loads(URLS_PATH.read_text(encoding="utf-8"))
    # de-dup by filename so encoded/unencoded duplicates collapse
    by_name: dict[str, str] = {}
    for u in urls:
        by_name[fname_of(u)] = u
    items = list(by_name.items())
    FILES.mkdir(parents=True, exist_ok=True)
    log(f"unique files={len(items)}")

    results = []
    pending: list[tuple[str, str, int | None]] = []
    for name, url in items:
        dest = FILES / name
        status, expected = head_info(url)
        time.sleep(0.05)
        if status in (404, 410):
            if dest.exists() and dest.stat().st_size < 50000:
                dest.unlink()
            results.append(
                {
                    "url": url,
                    "file": None,
                    "bytes": 0,
                    "expected": expected,
                    "ok": False,
                    "error": f"HTTP {status} missing on server",
                }
            )
            log(f"  404 {name}")
            continue
        if looks_complete(dest, expected):
            results.append(
                {
                    "url": url,
                    "file": str(dest),
                    "bytes": dest.stat().st_size,
                    "expected": expected,
                    "ok": True,
                    "cached": True,
                }
            )
            log(f"  have {dest.stat().st_size}/{expected or '?'} {name}")
        else:
            pending.append((name, url, expected))
            log(f"  need {dest.stat().st_size if dest.exists() else 0}/{expected or '?'} {name}")

    log(f"already_ok={len(results)} pending={len(pending)}")
    for i, (name, url, expected) in enumerate(pending, 1):
        log(f"[{i}/{len(pending)}] {name}")
        info = download(url, FILES / name, expected)
        results.append(info)
        status = "ok" if info["ok"] else "FAIL"
        log(f"  {status} {info.get('bytes')}/{expected or '?'} {name}")

    # second pass for remaining fails
    fails = [r for r in results if not r.get("ok")]
    if fails:
        log(f"second pass fails={len(fails)}")
        kept = [r for r in results if r.get("ok")]
        for r in fails:
            url = r["url"]
            name = fname_of(url)
            expected = r.get("expected")
            if str(r.get("error") or "").startswith("HTTP 404") or str(r.get("error") or "").startswith("HTTP 410"):
                kept.append(r)
                continue
            expected = expected or head_info(url)[1]
            info = download(url, FILES / name, expected)
            kept.append(info)
            log(f"  {'ok' if info['ok'] else 'FAIL'} {name}")
        results = kept

    ok_n = sum(1 for r in results if r.get("ok"))
    fail_n = len(results) - ok_n
    report = {
        "ok": ok_n,
        "fail": fail_n,
        "n": len(results),
        "bytes": sum(r.get("bytes") or 0 for r in results if r.get("ok")),
        "results": results,
    }
    (DATA / "downloads.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    missing = [r for r in results if "missing on server" in str(r.get("error") or "")]
    hard = [r for r in results if not r.get("ok") and r not in missing]
    log(f"DONE ok={ok_n} missing404={len(missing)} fail={len(hard)} bytes={report['bytes']}")
    if hard:
        for r in hard:
            log(f"  STILL {fname_of(r['url'])} {r.get('bytes')} {r.get('error')}")
        sys.exit(1)
    for r in missing:
        log(f"  404 {fname_of(r['url'])}")


if __name__ == "__main__":
    main()
