# -*- coding: utf-8 -*-
"""Build markdown/catalog from crawled raw JSON and download unique attachments in parallel."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
MD = DATA / "markdown"
FILES = DATA / "attachments" / "_files"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AIC2026Crawler/1.3"

PREFIX_RE = re.compile(r"^(2026AIC[·・.]?\s*|【赛马制】|【赛区制】)+")
ASSET_RE = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|zip|rar|7z)$", re.I)
PHONE_RE = re.compile(r"(?:手机号码|联系电话|电话)[：:]\s*([0-9\- ]{8,14})")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
QQ_RE = re.compile(r"(?:QQ群|交流QQ群|主题赛QQ群)[：:]\s*([^\n]{3,40})")
ORG_RE = re.compile(r"(?:承办单位|主办单位)[：:]\s*([^\n]{2,60})")


def log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()


def load_raw() -> list[dict]:
    rows = []
    for p in sorted(RAW.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("track") == "未分类":
            rec["skip_reason"] = "notice"
        rows.append(rec)
    return rows


def norm_title(title: str) -> str:
    t = (title or "").strip()
    t = PREFIX_RE.sub("", t)
    t = t.replace("赛题及竞赛规则", "").replace("赛题一览", "").replace("比赛规则", "")
    t = re.sub(r"\s+", "", t)
    t = t.replace("“", "").replace("”", "").replace("\"", "").replace("＋", "+")
    return t


def kind_of(rec: dict) -> str:
    title = rec.get("title") or ""
    cats = set(rec.get("categories") or [])
    if "一览" in title:
        return "overview"
    if "竞赛规则" in title or "赛题及竞赛" in title:
        return "rules"
    if cats & {48, 49, 50, 51, 52, 53}:
        return "nav"
    if "【赛马制】" in title:
        return "saima"
    if "【赛区制】" in title:
        return "saiqu"
    return "problem"


def excerpt(text: str, n: int = 180) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    return t[:n]


def collect_asset_urls(rec: dict) -> list[dict]:
    out = []
    seen = set()
    for lk in rec.get("links") or []:
        href = (lk.get("href") or "").strip()
        if href and ASSET_RE.search(urlparse(href).path) and href not in seen:
            seen.add(href)
            out.append({"text": lk.get("text") or "", "href": href})
    for im in rec.get("media") or []:
        href = (im.get("url") or "").strip()
        if href and ASSET_RE.search(urlparse(href).path) and href not in seen:
            seen.add(href)
            out.append({"text": im.get("title") or "", "href": href})
    return out


def write_markdown(rec: dict) -> Path:
    MD.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r'[\\/:*?"<>|]', "_", rec.get("title") or "untitled")[:50]
    path = MD / f"{rec['id']}_{safe}.md"
    assets = collect_asset_urls(rec)
    lines = [
        f"# {rec.get('title')}",
        "",
        f"- 赛道: {rec.get('track')}",
        f"- 类型: {kind_of(rec)}",
        f"- 发布: {rec.get('date')}",
        f"- 原文: {rec.get('link')}",
        f"- ID: {rec.get('id')}",
        "",
        "## 正文",
        "",
        rec.get("content_text") or "（空）",
        "",
        "## 文内链接",
        "",
    ]
    if rec.get("links"):
        for lk in rec["links"]:
            lines.append(f"- [{lk.get('text') or lk.get('href')}]({lk.get('href')})")
    else:
        lines.append("（无）")
    lines += ["", "## 附件链接", ""]
    if assets:
        for a in assets:
            lines.append(f"- [{a['text'] or a['href']}]({a['href']})")
    else:
        lines.append("（无）")
    path.write_text("\n".join(lines), encoding="utf-8")
    rec["markdown_file"] = str(path)
    rec["asset_urls"] = assets
    return path


def contacts(text: str) -> dict:
    emails = sorted(set(EMAIL_RE.findall(text or "")))
    phones = [m.group(1).strip() for m in PHONE_RE.finditer(text or "")]
    qqs = [m.group(1).strip() for m in QQ_RE.finditer(text or "")]
    orgs = [m.group(1).strip() for m in ORG_RE.finditer(text or "")]
    return {
        "emails": emails[:6],
        "phones": phones[:4],
        "qq": qqs[:4],
        "orgs": orgs[:4],
    }


def prefer(a: dict, b: dict) -> dict:
    """Keep the more complete of two duplicate pages."""
    ka, kb = kind_of(a), kind_of(b)
    rank = {"rules": 4, "saima": 3, "saiqu": 3, "problem": 2, "overview": 1, "nav": 0}
    if rank.get(ka, 0) != rank.get(kb, 0):
        return a if rank.get(ka, 0) > rank.get(kb, 0) else b
    la = len(a.get("content_text") or "")
    lb = len(b.get("content_text") or "")
    if "2026AIC" in (a.get("title") or "") and "2026AIC" not in (b.get("title") or ""):
        return a
    if "2026AIC" in (b.get("title") or "") and "2026AIC" not in (a.get("title") or ""):
        return b
    return a if la >= lb else b


def build_catalog(rows: list[dict]) -> dict:
    kept = [r for r in rows if r.get("skip_reason") != "notice"]
    by_track: dict[str, list] = {}
    for r in kept:
        by_track.setdefault(r.get("track") or "未分类", []).append(r)

    unique: dict[str, dict] = {}
    for r in kept:
        if kind_of(r) == "overview":
            continue
        key = (r.get("track"), norm_title(r.get("title") or ""))
        if key not in unique:
            unique[key] = r
        else:
            unique[key] = prefer(unique[key], r)

    unique_rows = list(unique.values())
    unique_by_track: dict[str, int] = {}
    for r in unique_rows:
        unique_by_track[r.get("track") or "未分类"] = unique_by_track.get(r.get("track") or "未分类", 0) + 1

    slim_all = []
    for r in sorted(kept, key=lambda x: (x.get("track") or "", x.get("date") or "", x.get("id") or 0)):
        ct = contacts(r.get("content_text") or "")
        slim_all.append(
            {
                "id": r["id"],
                "title": r.get("title"),
                "track": r.get("track"),
                "kind": kind_of(r),
                "date": (r.get("date") or "")[:10],
                "link": r.get("link"),
                "chars": len(r.get("content_text") or ""),
                "n_links": len(r.get("links") or []),
                "n_assets": len(r.get("asset_urls") or []),
                "assets": r.get("asset_urls") or [],
                "subproblems": r.get("subproblems") or [],
                "contacts": ct,
                "excerpt": excerpt(r.get("content_text") or ""),
                "markdown_file": r.get("markdown_file"),
            }
        )

    slim_unique = []
    for r in sorted(unique_rows, key=lambda x: (x.get("track") or "", x.get("title") or "")):
        ct = contacts(r.get("content_text") or "")
        slim_unique.append(
            {
                "id": r["id"],
                "title": r.get("title"),
                "norm": norm_title(r.get("title") or ""),
                "track": r.get("track"),
                "kind": kind_of(r),
                "date": (r.get("date") or "")[:10],
                "link": r.get("link"),
                "chars": len(r.get("content_text") or ""),
                "n_assets": len(r.get("asset_urls") or []),
                "assets": r.get("asset_urls") or [],
                "contacts": ct,
                "excerpt": excerpt(r.get("content_text") or "", 220),
            }
        )

    overviews = [s for s in slim_all if s["kind"] == "overview"]
    return {
        "source": "https://www.aicomp.cn/tracks",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "n_pages": len(kept),
        "n_unique_problems": len(unique_rows),
        "n_overviews": len(overviews),
        "by_track_pages": {k: len(v) for k, v in by_track.items()},
        "by_track_unique": unique_by_track,
        "overviews": overviews,
        "pages": slim_all,
        "problems": slim_unique,
    }


def download_one(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 2048:
        return {"url": url, "file": str(dest), "bytes": dest.stat().st_size, "ok": True, "cached": True}
    proc = subprocess.run(
        [
            "curl.exe",
            "-sS",
            "-L",
            "-C",
            "-",
            "--max-time",
            "180",
            "-A",
            UA,
            "-o",
            str(dest),
            url,
        ],
        capture_output=True,
        text=True,
        timeout=200,
    )
    size = dest.stat().st_size if dest.exists() else 0
    ok = proc.returncode in (0, 33) and size > 500
    return {
        "url": url,
        "file": str(dest) if dest.exists() else None,
        "bytes": size,
        "ok": ok,
        "error": None if ok else (proc.stderr.strip() or f"exit {proc.returncode} size={size}"),
    }


def fname_of(url: str) -> str:
    name = unquote(os.path.basename(urlparse(url).path) or "file")
    return re.sub(r'[\\/:*?"<>|]', "_", name) or "file"


def main() -> None:
    rows = load_raw()
    log(f"raw records={len(rows)}")
    for r in rows:
        write_markdown(r)
    catalog = build_catalog(rows)
    (DATA / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    (DATA / "index.json").write_text(
        json.dumps(
            {
                "source": catalog["source"],
                "built_at": catalog["built_at"],
                "n_pages": catalog["n_pages"],
                "n_unique_problems": catalog["n_unique_problems"],
                "by_track_pages": catalog["by_track_pages"],
                "by_track_unique": catalog["by_track_unique"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"pages={catalog['n_pages']} unique={catalog['n_unique_problems']}")
    log("by_track_unique " + json.dumps(catalog["by_track_unique"], ensure_ascii=False))

    urls = []
    seen = set()
    for r in rows:
        for a in r.get("asset_urls") or []:
            href = a["href"]
            if href not in seen:
                seen.add(href)
                urls.append(href)
    (DATA / "asset_urls.json").write_text(json.dumps(urls, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"unique assets={len(urls)}")

    FILES.mkdir(parents=True, exist_ok=True)
    results = []
    workers = 6
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_one, u, FILES / fname_of(u)): u for u in urls}
        done = 0
        for fut in as_completed(futs):
            info = fut.result()
            results.append(info)
            done += 1
            status = "ok" if info.get("ok") else "FAIL"
            log(f"  [{done}/{len(urls)}] {status} {info.get('bytes')} {Path(info.get('file') or info['url']).name}")
    (DATA / "downloads.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ok_n = sum(1 for x in results if x.get("ok"))
    fail_n = len(results) - ok_n
    log(f"downloads ok={ok_n} fail={fail_n}")
    catalog["downloads"] = {"ok": ok_n, "fail": fail_n, "n": len(results)}
    (DATA / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
