# -*- coding: utf-8 -*-
"""Crawl 2026 AIC tracks and problems from aicomp.cn."""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path

BASE = "https://www.aicomp.cn"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AIC2026Crawler/1.2"
SLEEP = 0.05

# 赛道赛题 + 子类；aic-1..aic-6 为当年赛题导航
TRACK_CAT_IDS = {3, 4, 5, 6, 23, 24, 36}
NAV_CAT_IDS = {48, 49, 50, 51, 52, 53}
ALL_CAT_IDS = TRACK_CAT_IDS | NAV_CAT_IDS

CAT_NAMES = {
    3: "赛道赛题",
    4: "算法挑战赛",
    5: "算法创新赛",
    6: "算法应用赛",
    23: "算法主题赛",
    24: "产业命题赛",
    36: "算法专项赛",
    48: "算法挑战赛",
    49: "算法创新赛",
    50: "算法应用赛",
    51: "算法主题赛",
    52: "产业命题赛",
    53: "算法专项赛",
}

ASSET_EXT = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|zip|rar|7z)$", re.I)
POST_ID_RE = re.compile(r"/(\d+)\.html(?:$|[?#])")
FIELDS_LIST = "id,date,modified,link,title,categories"
FIELDS_FULL = "id,date,modified,link,title,categories,content,excerpt"


def log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write((msg + "\n").encode(enc, errors="replace"))
        sys.stdout.buffer.flush()


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[dict] = []
        self.images: list[dict] = []
        self.texts: list[str] = []
        self._skip = 0
        self._in_a = False
        self._href = ""
        self._a_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag in ("script", "style"):
            self._skip += 1
            return
        if tag == "br":
            self.texts.append("\n")
        elif tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self.texts.append("\n")
        if tag == "a":
            self._in_a = True
            self._href = d.get("href", "")
            self._a_text = []
        if tag == "img":
            src = d.get("src") or d.get("data-src") or ""
            if src:
                self.images.append({"src": src, "alt": d.get("alt", "")})

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
            return
        if tag == "a" and self._in_a:
            text = "".join(self._a_text).strip()
            if self._href:
                self.links.append({"href": self._href, "text": text})
            self._in_a = False
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self.texts.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_a:
            self._a_text.append(data)
        self.texts.append(data)


def http_get(url: str, retries: int = 3) -> tuple[dict, bytes]:
    last: Exception | None = None
    for i in range(retries):
        hdr_path = None
        body_path = None
        try:
            fd_h, hdr_path = tempfile.mkstemp(prefix="aic-h-", suffix=".txt")
            fd_b, body_path = tempfile.mkstemp(prefix="aic-b-", suffix=".bin")
            os.close(fd_h)
            os.close(fd_b)
            proc = subprocess.run(
                [
                    "curl.exe",
                    "-sS",
                    "-L",
                    "--max-time",
                    "30",
                    "-A",
                    UA,
                    "-D",
                    hdr_path,
                    "-o",
                    body_path,
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=40,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or f"curl exit {proc.returncode}")
            raw_headers = Path(hdr_path).read_text(encoding="utf-8", errors="replace")
            headers: dict[str, str] = {}
            for line in raw_headers.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            body = Path(body_path).read_bytes()
            return headers, body
        except Exception as e:  # noqa: BLE001
            last = e
            log(f"  retry {i+1} {url} ({e})")
            time.sleep(0.4 * (i + 1))
        finally:
            for p in (hdr_path, body_path):
                if p:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
    raise RuntimeError(f"GET failed {url}: {last}")


def wp_json(url: str) -> tuple[dict, object]:
    headers, body = http_get(url)
    return headers, json.loads(body.decode("utf-8"))


def decode_title(t: str) -> str:
    t = html.unescape(t or "").replace("\xa0", " ")
    return re.sub(r"\s+", " ", t).strip()


def slugify(s: str, maxlen: int = 50) -> str:
    s = decode_title(s)
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    s = re.sub(r"\s+", "_", s)
    return (s[:maxlen].strip("._") or "untitled")


def abs_url(href: str, base: str = BASE) -> str:
    href = html.unescape((href or "").strip())
    if not href or href.lower().startswith(("mailto:", "javascript:", "tel:", "#")):
        return ""
    return urllib.parse.urljoin(base.rstrip("/") + "/", href)


def parse_html(raw: str) -> dict:
    p = PageParser()
    p.feed(raw or "")
    text = html.unescape("".join(p.texts))
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return {"text": text, "links": p.links, "images": p.images}


def post_id_from_url(url: str) -> int | None:
    m = POST_ID_RE.search(url or "")
    return int(m.group(1)) if m else None


def is_internal_post(url: str) -> bool:
    if not url:
        return False
    u = urllib.parse.urlparse(url)
    if u.netloc and u.netloc not in ("www.aicomp.cn", "aicomp.cn"):
        return False
    return bool(POST_ID_RE.search(u.path or ""))


def is_asset(url: str) -> bool:
    path = urllib.parse.urlparse(url).path
    if ASSET_EXT.search(path):
        return True
    if "/wp-content/uploads/" in path and ASSET_EXT.search(path):
        return True
    return bool(ASSET_EXT.search(path))


def track_label(cats: list[int]) -> str:
    for cid in (4, 5, 6, 23, 24, 36, 48, 49, 50, 51, 52, 53):
        if cid in cats:
            return CAT_NAMES[cid]
    if 3 in cats:
        return "赛道赛题"
    return "未分类"


def fetch_list(cat_ids: set[int]) -> dict[int, dict]:
    posts: dict[int, dict] = {}
    for cid in sorted(cat_ids):
        page = 1
        while True:
            url = (
                f"{BASE}/wp-json/wp/v2/posts?categories={cid}"
                f"&per_page=50&page={page}&_fields={FIELDS_LIST}"
            )
            try:
                headers, data = wp_json(url)
            except Exception as e:
                log(f"WARN skip cat={cid} page={page}: {e}")
                break
            if not data:
                break
            for p in data:
                pid = p["id"]
                if pid not in posts:
                    posts[pid] = p
                else:
                    cats = set(posts[pid].get("categories") or []) | set(p.get("categories") or [])
                    posts[pid]["categories"] = sorted(cats)
            total_pages = int(headers.get("x-wp-totalpages") or "1")
            log(f"list cat={cid} page={page}/{total_pages} +{len(data)} unique={len(posts)}")
            if page >= total_pages:
                break
            page += 1
            time.sleep(SLEEP)
    return posts


def search_list(term: str) -> dict[int, dict]:
    posts: dict[int, dict] = {}
    page = 1
    while True:
        q = urllib.parse.quote(term)
        url = (
            f"{BASE}/wp-json/wp/v2/posts?search={q}"
            f"&per_page=100&page={page}&_fields={FIELDS_LIST}"
        )
        try:
            headers, data = wp_json(url)
        except Exception as e:
            log(f"WARN search {term}: {e}")
            break
        if not data:
            break
        for p in data:
            posts[p["id"]] = p
        total_pages = int(headers.get("x-wp-totalpages") or "1")
        log(f"search {term!r} page={page}/{total_pages} +{len(data)}")
        if page >= total_pages:
            break
        page += 1
        time.sleep(SLEEP)
    return posts


def fetch_full(pid: int) -> dict | None:
    url = f"{BASE}/wp-json/wp/v2/posts/{pid}?_fields={FIELDS_FULL}"
    try:
        _, data = wp_json(url)
        return data if isinstance(data, dict) else None
    except Exception as e:
        log(f"WARN full {pid}: {e}")
        return None


def fetch_media(pid: int) -> list[dict]:
    url = f"{BASE}/wp-json/wp/v2/media?parent={pid}&per_page=100&_fields=id,source_url,mime_type,title,alt_text,slug"
    try:
        _, data = wp_json(url)
        return data if isinstance(data, list) else []
    except Exception as e:
        log(f"WARN media {pid}: {e}")
        return []


def listing_pages_ids() -> list[tuple[int, str, str]]:
    """Parse /tracks/ HTML pagination for publicly listed items."""
    found: list[tuple[int, str, str]] = []
    seen: set[int] = set()
    for page in range(1, 8):
        url = f"{BASE}/tracks/" if page == 1 else f"{BASE}/tracks/page/{page}"
        try:
            _, body = http_get(url)
        except Exception as e:
            log(f"WARN listing {url}: {e}")
            break
        html_text = body.decode("utf-8", errors="ignore")
        # typical: href=".../1234.html" ... title
        for m in re.finditer(
            r'href="(https://www\.aicomp\.cn/[^"]+?/(\d+)\.html)"[^>]*>[\s\S]{0,200}?<h[1-6][^>]*>([\s\S]*?)</h[1-6]>',
            html_text,
            re.I,
        ):
            pid = int(m.group(2))
            title = decode_title(re.sub(r"<[^>]+>", "", m.group(3)))
            if pid not in seen:
                seen.add(pid)
                found.append((pid, title, m.group(1)))
        if f"/tracks/page/{page + 1}" not in html_text and page > 1:
            # last page if no next
            if "下一页" not in html_text and f"page/{page + 1}" not in html_text:
                break
        time.sleep(SLEEP)
    log(f"html listing items={len(found)}")
    return found


def looks_2026(meta: dict, listing_2026_ids: set[int] | None = None) -> bool:
    title = decode_title(meta.get("title", {}).get("rendered", ""))
    cats = set(meta.get("categories") or [])
    date = (meta.get("date") or "")[:10]
    pid = meta.get("id")
    if listing_2026_ids and pid in listing_2026_ids:
        return True
    if date and date < "2026-01-01":
        return False
    if cats & (NAV_CAT_IDS | TRACK_CAT_IDS) and date >= "2026-01-01":
        return True
    if cats & TRACK_CAT_IDS and "2026" in title:
        return True
    return False


def to_record(post: dict) -> dict:
    title = decode_title(post.get("title", {}).get("rendered", ""))
    content_html = (post.get("content") or {}).get("rendered") or ""
    parsed = parse_html(content_html)
    cats = list(post.get("categories") or [])
    rec = {
        "id": post.get("id"),
        "title": title,
        "date": post.get("date"),
        "modified": post.get("modified"),
        "link": post.get("link"),
        "categories": cats,
        "track": track_label(cats),
        "content_html": content_html,
        "content_text": parsed["text"],
        "links": [
            {"text": decode_title(lk.get("text", "")), "href": abs_url(lk.get("href", ""), post.get("link") or BASE)}
            for lk in parsed["links"]
            if abs_url(lk.get("href", ""), post.get("link") or BASE)
        ],
        "images": [
            {"src": abs_url(im.get("src", ""), post.get("link") or BASE), "alt": im.get("alt", "")}
            for im in parsed["images"]
        ],
        "media": [],
        "downloaded": [],
        "is_2026": True,
        "source": "wp_post",
    }
    return rec


def download_asset(url: str, dest_dir: Path, hint: str = "", cache: dict | None = None) -> dict:
    url = abs_url(url)
    if cache is not None and url in cache:
        return dict(cache[url])
    parsed = urllib.parse.urlparse(url)
    fname = urllib.parse.unquote(os.path.basename(parsed.path) or "file")
    fname = re.sub(r'[\\/:*?"<>|]', "_", fname)
    if hint and "." not in fname:
        fname = slugify(hint) + "_" + fname
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / fname
    if dest.exists() and dest.stat().st_size > 2048:
        info = {"url": url, "file": str(dest), "bytes": dest.stat().st_size, "name": dest.name, "cached": True}
        log(f"  have {dest.name} ({info['bytes']} bytes)")
        if cache is not None:
            cache[url] = info
        return info
    last_err = None
    for i in range(4):
        try:
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
            if proc.returncode not in (0, 33):  # 33 = resume beyond file size / already complete
                raise RuntimeError(proc.stderr.strip() or f"curl exit {proc.returncode}")
            size = dest.stat().st_size if dest.exists() else 0
            if size < 500:
                raise RuntimeError(f"too small {size} bytes")
            info = {"url": url, "file": str(dest), "bytes": size, "name": dest.name}
            log(f"  saved {dest.name} ({size} bytes)")
            if cache is not None:
                cache[url] = info
            return info
        except Exception as e:  # noqa: BLE001
            last_err = e
            log(f"  retry-dl {i+1} {fname} ({e})")
            time.sleep(0.8 * (i + 1))
    info = {"url": url, "file": str(dest) if dest.exists() else None, "error": str(last_err), "name": fname}
    log(f"  FAIL {url}: {last_err}")
    if cache is not None:
        cache[url] = info
    return info


def extract_subproblems(text: str, links: list[dict]) -> list[dict]:
    items: list[dict] = []
    # 赛题一、xxx / 赛题1： / 题目一
    for m in re.finditer(
        r"(赛题[一二三四五六七八九十百零0-9]+[、.:：]\s*[^\n]{2,80})",
        text or "",
    ):
        items.append({"heading": m.group(1).strip()})
    # de-dup headings
    seen = set()
    out = []
    for it in items:
        if it["heading"] not in seen:
            seen.add(it["heading"])
            out.append(it)
    return out


def write_markdown(rec: dict, md_dir: Path) -> Path:
    md_dir.mkdir(parents=True, exist_ok=True)
    path = md_dir / f"{rec['id']}_{slugify(rec['title'])}.md"
    lines = [
        f"# {rec['title']}",
        "",
        f"- 赛道: {rec['track']}",
        f"- 发布: {rec['date']}",
        f"- 更新: {rec['modified']}",
        f"- 原文: {rec['link']}",
        f"- ID: {rec['id']}",
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
    lines += ["", "## 附件", ""]
    if rec.get("downloaded"):
        for d in rec["downloaded"]:
            lines.append(f"- {d.get('name')} ({d.get('bytes')} bytes) {d.get('url')}")
    else:
        lines.append("（无已下载附件）")
    if rec.get("subproblems"):
        lines += ["", "## 解析出的子赛题标题", ""]
        for s in rec["subproblems"]:
            lines.append(f"- {s['heading']}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def crawl(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    md_dir = out_dir / "markdown"
    asset_dir = out_dir / "attachments"
    for d in (raw_dir, md_dir, asset_dir):
        d.mkdir(exist_ok=True)

    log("== phase1 list posts ==")
    posts = fetch_list(ALL_CAT_IDS)
    listed_path = out_dir / "listed.json"
    listed_rows = []
    for pid, m in sorted(posts.items()):
        listed_rows.append(
            {
                "id": pid,
                "title": decode_title(m.get("title", {}).get("rendered", "")),
                "date": m.get("date"),
                "link": m.get("link"),
                "categories": m.get("categories") or [],
                "track": track_label(list(m.get("categories") or [])),
                "is_2026": looks_2026(m),
            }
        )
    listed_path.write_text(json.dumps(listed_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    seed = {row["id"] for row in listed_rows if row["is_2026"]}
    log(f"listed posts={len(posts)} seed_2026={len(seed)} wrote {listed_path}")
    if os.environ.get("LIST_ONLY") == "1":
        log("LIST_ONLY=1 stop after index")
        return {"n_listed": len(posts), "n_2026": len(seed)}

    log("== phase2 fetch full content ==")
    records: dict[int, dict] = {}
    queue = list(seed)
    seen_fetch: set[int] = set()
    while queue:
        pid = queue.pop(0)
        if pid in seen_fetch:
            continue
        seen_fetch.add(pid)
        cached = raw_dir / f"{pid}.json"
        rec = None
        if cached.exists():
            try:
                rec = json.loads(cached.read_text(encoding="utf-8"))
                log(f"  cache {pid} {str(rec.get('title',''))[:70]}")
            except Exception:
                rec = None
        if rec is None:
            full = fetch_full(pid)
            time.sleep(SLEEP)
            if not full:
                meta = posts.get(pid) or {"id": pid, "title": {"rendered": str(pid)}, "categories": []}
                rec = to_record({**meta, "content": {"rendered": ""}})
                rec["fetch_error"] = True
                records[pid] = rec
                continue
            rec = to_record(full)
            rec["is_2026"] = True
            rec["subproblems"] = extract_subproblems(rec["content_text"], rec.get("links") or [])
            (raw_dir / f"{pid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            log(f"  got {pid} [{rec['track']}] {rec['title'][:70]}")
        rec["is_2026"] = True
        rec.setdefault("links", [])
        records[pid] = rec
        for lk in rec.get("links") or []:
            href = lk.get("href") or ""
            nid = post_id_from_url(href)
            if not (nid and is_internal_post(href) and nid not in seen_fetch):
                continue
            path = urllib.parse.urlparse(href).path or ""
            text = lk.get("text") or ""
            # only follow other 赛题 pages, not notices/news
            if "/tracks/" in path or "/aic-" in path or "赛题" in text or "规则" in text:
                queue.append(nid)
                if nid not in posts:
                    posts[nid] = {
                        "id": nid,
                        "title": {"rendered": text or str(nid)},
                        "categories": [],
                        "link": href,
                    }

    log(f"fetched 2026 records={len(records)}")

    log("== phase3 media + attachments ==")
    asset_cache: dict[str, dict] = {}
    files_dir = asset_dir / "_files"
    files_dir.mkdir(exist_ok=True)
    for rec in records.values():
        pid = rec["id"]
        rec["downloaded"] = []
        rec["media"] = []
        try:
            media = fetch_media(pid)
        except Exception as e:
            log(f"WARN media {pid}: {e}")
            media = []
        time.sleep(SLEEP)
        rec["media"] = []
        seen_asset: set[str] = set()
        for m in media:
            src = m.get("source_url") or ""
            title = decode_title((m.get("title") or {}).get("rendered") or m.get("slug") or "")
            item = {"id": m.get("id"), "title": title, "mime": m.get("mime_type"), "url": src}
            rec["media"].append(item)
            if src and is_asset(src) and src not in seen_asset:
                info = download_asset(src, files_dir, title, cache=asset_cache)
                rec["downloaded"].append(info)
                seen_asset.add(src)
        for lk in rec.get("links") or []:
            href = lk.get("href") or ""
            if href in seen_asset:
                continue
            if is_asset(href):
                info = download_asset(href, files_dir, lk.get("text") or "", cache=asset_cache)
                rec["downloaded"].append({**lk, **info})
                seen_asset.add(href)
        md_path = write_markdown(rec, md_dir)
        rec["markdown_file"] = str(md_path)
        (raw_dir / f"{pid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    by_track: dict[str, list] = {}
    for rec in sorted(records.values(), key=lambda r: (r["track"], r["date"] or "", r["id"])):
        by_track.setdefault(rec["track"], []).append(
            {
                "id": rec["id"],
                "title": rec["title"],
                "date": rec["date"],
                "link": rec["link"],
                "track": rec["track"],
                "n_links": len(rec.get("links") or []),
                "n_assets": len(rec.get("downloaded") or []),
                "subproblems": rec.get("subproblems") or [],
                "markdown_file": rec.get("markdown_file"),
                "excerpt": (rec.get("content_text") or "").replace("\n", " ")[:200],
            }
        )

    excluded = []
    for pid, meta in posts.items():
        if pid not in records:
            excluded.append(
                {
                    "id": pid,
                    "title": decode_title(meta.get("title", {}).get("rendered", "")),
                    "date": meta.get("date"),
                    "link": meta.get("link"),
                    "track": track_label(list(meta.get("categories") or [])),
                }
            )

    index = {
        "source": BASE + "/tracks",
        "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "n_listed": len(posts),
        "n_2026": len(records),
        "n_excluded": len(excluded),
        "by_track_counts": {k: len(v) for k, v in by_track.items()},
        "by_track": by_track,
        "excluded_non_2026": excluded,
        "html_listing": [],
    }
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    catalog = []
    for rec in sorted(records.values(), key=lambda r: (r["track"], r["date"] or "", r["id"])):
        catalog.append(
            {
                "id": rec["id"],
                "title": rec["title"],
                "track": rec["track"],
                "date": rec["date"],
                "modified": rec["modified"],
                "link": rec["link"],
                "categories": rec["categories"],
                "content_text": rec.get("content_text"),
                "links": rec.get("links"),
                "subproblems": rec.get("subproblems"),
                "downloaded": rec.get("downloaded"),
                "media": rec.get("media"),
                "markdown_file": rec.get("markdown_file"),
            }
        )
    (out_dir / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    log("DONE")
    log("counts " + json.dumps(index["by_track_counts"], ensure_ascii=False))
    return index


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "data"))
    crawl(out)
