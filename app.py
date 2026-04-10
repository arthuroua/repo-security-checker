import asyncio
import json
import os
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import requests
from flask import Flask, jsonify, render_template, request

try:
    import opengradient as og
except Exception:
    og = None


app = Flask(__name__)

PRODUCT_NAME = "OpenGradient Repo Risk Radar"
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "40"))
GITHUB_API_BASE = os.getenv("GITHUB_API_BASE", "https://api.github.com")
REPO_CHECK_README_MAX_CHARS = int(os.getenv("REPO_CHECK_README_MAX_CHARS", "5000"))

INFERENCE_PROVIDER = os.getenv("INFERENCE_PROVIDER", "opengradient_sdk").strip().lower()
OG_PRIVATE_KEY = os.getenv("OG_PRIVATE_KEY")
OG_SDK_MODEL = os.getenv("OG_SDK_MODEL", "GEMINI_2_5_FLASH")
OG_SETTLEMENT_MODE = os.getenv("OG_SETTLEMENT_MODE", "PRIVATE").upper()
OG_APPROVAL_OPG_AMOUNT = float(os.getenv("OG_APPROVAL_OPG_AMOUNT", "5"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a security analyst. Be practical, specific, and concise.",
)


def _json_error(message: str, status: int = 500, details: Any | None = None):
    payload = {"error": message}
    if details is not None:
        payload["details"] = details
    return jsonify(payload), status


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "OpenGradientRepoSecurityChecker/1.0",
    }
    github_token = (os.getenv("GITHUB_TOKEN") or "").strip()
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


def _parse_github_repo_input(repo_input: str) -> tuple[str, str, str]:
    value = (repo_input or "").strip()
    if not value:
        raise ValueError("GitHub repository URL is required")

    value = value.replace("git@github.com:", "https://github.com/").replace(".git", "")
    if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", value):
        owner, repo = value.split("/", 1)
        return owner, repo, f"https://github.com/{owner}/{repo}"

    parsed = urlparse(value)
    if parsed.netloc.lower() not in ("github.com", "www.github.com"):
        raise ValueError("Only github.com repositories are supported")

    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError("Invalid GitHub repository URL. Expected format: https://github.com/owner/repo")
    owner, repo = parts[0], parts[1]
    return owner, repo, f"https://github.com/{owner}/{repo}"


def _fetch_repo_snapshot(owner: str, repo: str) -> dict[str, Any]:
    headers = _github_headers()

    repo_resp = requests.get(f"{GITHUB_API_BASE}/repos/{owner}/{repo}", headers=headers, timeout=REQUEST_TIMEOUT)
    if repo_resp.status_code == 404:
        raise RuntimeError("Repository not found or private repository is inaccessible")
    repo_resp.raise_for_status()
    repo_data = repo_resp.json()

    readme_text = ""
    readme_resp = requests.get(f"{GITHUB_API_BASE}/repos/{owner}/{repo}/readme", headers=headers, timeout=REQUEST_TIMEOUT)
    if readme_resp.status_code == 200:
        payload = readme_resp.json()
        content = (payload.get("content") or "").replace("\n", "")
        if content:
            try:
                import base64

                readme_text = base64.b64decode(content).decode("utf-8", errors="ignore")
            except Exception:
                readme_text = ""

    langs = {}
    langs_resp = requests.get(f"{GITHUB_API_BASE}/repos/{owner}/{repo}/languages", headers=headers, timeout=REQUEST_TIMEOUT)
    if langs_resp.status_code == 200:
        try:
            langs = langs_resp.json()
        except Exception:
            langs = {}

    return {
        "repo": {
            "full_name": repo_data.get("full_name"),
            "html_url": repo_data.get("html_url"),
            "description": repo_data.get("description"),
            "default_branch": repo_data.get("default_branch"),
            "created_at": repo_data.get("created_at"),
            "updated_at": repo_data.get("updated_at"),
            "pushed_at": repo_data.get("pushed_at"),
            "stargazers_count": int(repo_data.get("stargazers_count") or 0),
            "forks_count": int(repo_data.get("forks_count") or 0),
            "open_issues_count": int(repo_data.get("open_issues_count") or 0),
            "watchers_count": int(repo_data.get("subscribers_count") or repo_data.get("watchers_count") or 0),
            "archived": bool(repo_data.get("archived")),
            "disabled": bool(repo_data.get("disabled")),
            "license": (repo_data.get("license") or {}).get("spdx_id"),
            "topics": repo_data.get("topics") or [],
        },
        "languages": langs,
        "readme_excerpt": (readme_text or "")[:REPO_CHECK_README_MAX_CHARS],
    }


def _safe_parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _build_repo_heuristics(snapshot: dict[str, Any]) -> dict[str, Any]:
    repo = snapshot.get("repo") or {}
    readme = (snapshot.get("readme_excerpt") or "").lower()
    findings: list[str] = []
    risk_score = 0

    if repo.get("archived"):
        risk_score += 30
        findings.append("Repository is archived (maintenance risk).")
    if repo.get("disabled"):
        risk_score += 40
        findings.append("Repository is disabled by GitHub.")
    if not repo.get("license") or repo.get("license") in ("NOASSERTION", "NONE", None):
        risk_score += 10
        findings.append("No clear license detected.")
    if not (snapshot.get("readme_excerpt") or "").strip():
        risk_score += 15
        findings.append("README is missing or inaccessible.")

    created_at = _safe_parse_iso8601(repo.get("created_at"))
    pushed_at = _safe_parse_iso8601(repo.get("pushed_at"))
    now_utc = datetime.now(timezone.utc)
    stars = int(repo.get("stargazers_count") or 0)

    if created_at and (now_utc - created_at).days < 14 and stars < 3:
        risk_score += 15
        findings.append("Very new repository with low social proof.")
    if pushed_at and (now_utc - pushed_at).days > 365:
        risk_score += 10
        findings.append("No recent commits in over 1 year.")

    suspicious_patterns = {
        "curl pipe shell command in README": r"curl[^\n]*\|[^\n]*(sh|bash)",
        "wget pipe shell command in README": r"wget[^\n]*\|[^\n]*(sh|bash)",
        "private key or seed phrase mention": r"private[_ -]?key|seed phrase|mnemonic",
        "disabled ssl verification mention": r"verify\s*=\s*false|--insecure|ssl\s*verify\s*false",
    }
    for label, pattern in suspicious_patterns.items():
        if re.search(pattern, readme):
            risk_score += 10
            findings.append(label)

    risk_score = max(0, min(risk_score, 100))
    if risk_score >= 75:
        verdict = "critical"
    elif risk_score >= 50:
        verdict = "high"
    elif risk_score >= 25:
        verdict = "medium"
    else:
        verdict = "low"

    return {"risk_score": risk_score, "verdict": verdict, "findings": findings}


def _build_repo_security_prompt(snapshot: dict[str, Any], heuristics: dict[str, Any], focus: str) -> str:
    return (
        "You are a senior security auditor.\n"
        f"Focus requested by user: {focus or 'general repository security'}.\n\n"
        "Repository metadata:\n"
        f"{json.dumps(snapshot.get('repo') or {}, ensure_ascii=False, indent=2)}\n\n"
        "Languages:\n"
        f"{json.dumps(snapshot.get('languages') or {}, ensure_ascii=False, indent=2)}\n\n"
        "Heuristic pre-scan:\n"
        f"{json.dumps(heuristics, ensure_ascii=False, indent=2)}\n\n"
        "README excerpt:\n"
        f"{(snapshot.get('readme_excerpt') or '')[:3500]}\n\n"
        "Return concise markdown with:\n"
        "1) Verdict (Low/Medium/High/Critical)\n"
        "2) Top risks (max 5)\n"
        "3) What to verify manually next\n"
        "4) Recommendation for users before trusting this repo\n"
        "Do not hallucinate. If data is missing, say it clearly."
    )


def _resolve_og_model():
    if og is None:
        raise RuntimeError("OpenGradient SDK not installed")
    model = getattr(og.TEE_LLM, OG_SDK_MODEL, None)
    if model is None:
        raise RuntimeError(f"Unsupported OG_SDK_MODEL: {OG_SDK_MODEL}")
    return model


def _resolve_settlement_mode():
    if og is None:
        raise RuntimeError("OpenGradient SDK not installed")
    mode = getattr(og.x402SettlementMode, OG_SETTLEMENT_MODE, None)
    if mode is None:
        return og.x402SettlementMode.PRIVATE
    return mode


async def _call_og_sdk_async(prompt: str) -> str:
    if og is None:
        raise RuntimeError("OpenGradient SDK unavailable")
    if not OG_PRIVATE_KEY:
        raise RuntimeError("OG_PRIVATE_KEY is not set")

    llm = og.LLM(private_key=OG_PRIVATE_KEY)
    errors = []
    for kwargs in (
        {"min_allowance": OG_APPROVAL_OPG_AMOUNT},
        {"opg_amount": OG_APPROVAL_OPG_AMOUNT},
        {},
    ):
        try:
            llm.ensure_opg_approval(**kwargs)
            break
        except TypeError as exc:
            errors.append(str(exc))
            continue
    else:
        raise RuntimeError("ensure_opg_approval failed for all known signatures: " + " | ".join(errors))
    result = await llm.chat(
        model=_resolve_og_model(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=600,
        x402_settlement_mode=_resolve_settlement_mode(),
    )
    content = result.chat_output.get("content") if isinstance(result.chat_output, dict) else str(result.chat_output)
    return (content or "").strip()


def _call_og_sdk(prompt: str) -> str:
    return _run_async(_call_og_sdk_async(prompt))


def _call_openai(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 600,
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT}\n\n{prompt}"}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 700},
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def run_llm(prompt: str) -> tuple[str, str]:
    provider = INFERENCE_PROVIDER
    if provider == "opengradient_sdk":
        return _call_og_sdk(prompt), "opengradient_sdk"
    if provider == "openai":
        return _call_openai(prompt), "openai"
    if provider == "gemini":
        return _call_gemini(prompt), "gemini"

    errors = []
    for name, fn in (("opengradient_sdk", _call_og_sdk), ("openai", _call_openai), ("gemini", _call_gemini)):
        try:
            return fn(prompt), name
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("All inference providers failed: " + " | ".join(errors))


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "product": PRODUCT_NAME,
            "provider": INFERENCE_PROVIDER,
            "has_og_private_key": bool(OG_PRIVATE_KEY),
            "has_openai_key": bool(OPENAI_API_KEY),
            "has_gemini_key": bool(GEMINI_API_KEY),
            "og_sdk_available": og is not None,
        }
    )


@app.post("/api/repo-security/check")
def repo_security_check():
    data = request.get_json(silent=True) or {}
    repo_input = (data.get("repo_url") or data.get("repo") or "").strip()
    focus = (data.get("focus") or "smart-contract and dependency risks").strip()

    try:
        owner, repo, canonical_url = _parse_github_repo_input(repo_input)
        snapshot = _fetch_repo_snapshot(owner, repo)
        heuristics = _build_repo_heuristics(snapshot)
        prompt = _build_repo_security_prompt(snapshot, heuristics, focus)
        analysis, provider = run_llm(prompt)
        return jsonify(
            {
                "ok": True,
                "repo": canonical_url,
                "metadata": snapshot.get("repo") or {},
                "languages": snapshot.get("languages") or {},
                "heuristics": heuristics,
                "analysis": analysis,
                "provider": provider,
            }
        )
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except requests.HTTPError as exc:
        details = exc.response.text[:500] if exc.response is not None else str(exc)
        return _json_error("GitHub API error", 502, details)
    except requests.RequestException as exc:
        return _json_error("Network error while fetching repository", 502, str(exc))
    except Exception as exc:
        return _json_error("Repo security check failed", 500, str(exc))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
