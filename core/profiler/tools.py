"""Bounded, source-linked company research with outbound SSRF protection."""

import asyncio
import ipaddress
import hashlib
from html.parser import HTMLParser
import logging
import socket
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.runtime.agent import sanitize_secrets
from core.runtime.tooling import RuntimeTool

logger = logging.getLogger(__name__)


class _ReadableHTML(HTMLParser):
    """Extract a compact text view while discarding active and non-content elements."""

    _DROP = {"script", "style", "noscript", "svg", "iframe", "canvas", "template", "nav", "footer"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._drop_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag in self._DROP:
            self._drop_depth += 1
        elif tag in {"p", "br", "li", "h1", "h2", "h3", "h4", "tr", "section", "article"} and not self._drop_depth:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._DROP and self._drop_depth:
            self._drop_depth -= 1
        elif tag in {"p", "li", "h1", "h2", "h3", "h4", "tr", "section", "article"} and not self._drop_depth:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._drop_depth and data.strip():
            self.parts.append(data.strip())


def _readable_page_text(content: bytes, content_type: str) -> str:
    if "text/plain" in content_type:
        raw = content.decode("utf-8", errors="replace")
    elif "text/html" in content_type or not content_type:
        parser = _ReadableHTML()
        parser.feed(content.decode("utf-8", errors="replace"))
        raw = " ".join(parser.parts)
    else:
        return ""
    return " ".join(raw.split())[:8_000]

# RFC 6598 Carrier-Grade NAT network
CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

# Disallowed internal domain suffixes and hostnames
DISALLOWED_HOSTS = {
    "localhost",
    "instance-data",
    "metadata.google.internal",
}

DISALLOWED_DOMAIN_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".corp",
    ".lan",
    ".home",
    ".test",
    ".invalid",
    ".example",
)


def is_ip_disallowed(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Evaluate whether an IP address belongs to private, loopback, link-local or reserved ranges.

    Args:
        ip: Evaluated IPv4 or IPv6 address object.

    Returns:
        bool: True if IP is unsafe for outbound egress requests, False if public.
    """
    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        return True

    # Carrier-grade NAT (100.64.0.0/10)
    if isinstance(ip, ipaddress.IPv4Address) and ip in CGNAT_NETWORK:
        return True

    return False


def validate_outbound_url(url: str, allow_http: bool = False) -> bool:
    """Validate an outbound URL against SSRF attacks, internal network scanning, and loopbacks.

    Enforces:
    1. Scheme check: HTTPS only by default (rejects file://, gopher://, javascript:, etc.).
    2. Hostname validation: Rejects loopback/private/internal hostnames.
    3. Literal IP address evaluation: Rejects loopback, RFC 1918 private, link-local (169.254.169.254),
       IPv6 ::1, unique local addresses (fc00::/7), and CGNAT.
    4. DNS resolution: Resolves domain to IP addresses and evaluates every resolved target. Fail-closed.

    Args:
        url: Outbound candidate URL string.
        allow_http: Optional override to allow unencrypted http scheme (defaults to False).

    Returns:
        bool: True if safe for outbound request, False if blocked.
    """
    if not isinstance(url, str) or not url.strip():
        return False

    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception:
        return False

    # 1. Scheme check
    allowed_schemes = ("https", "http") if allow_http else ("https",)
    if not parsed.scheme or parsed.scheme.lower() not in allowed_schemes:
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    clean_host = hostname.strip("[]").lower()

    # 2. Block known local / metadata hostnames and suffixes
    if clean_host in DISALLOWED_HOSTS:
        return False

    for suffix in DISALLOWED_DOMAIN_SUFFIXES:
        if clean_host.endswith(suffix):
            return False

    # 3. Check direct IP address literal
    try:
        ip = ipaddress.ip_address(clean_host)
        return not is_ip_disallowed(ip)
    except ValueError:
        pass  # Hostname is a domain name, proceed to DNS resolution

    # 4. Resolve domain name via DNS and inspect resolved IPs
    try:
        # Use port 443 or default if unspecified
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
        addr_infos = socket.getaddrinfo(clean_host, port, proto=socket.IPPROTO_TCP)
        if not addr_infos:
            return False

        # Một địa chỉ DNS bị cấm là đủ từ chối hostname, kể cả các địa chỉ còn lại công khai.
        for addr in addr_infos:
            ip_str = addr[4][0]
            resolved_ip = ipaddress.ip_address(ip_str)
            if is_ip_disallowed(resolved_ip):
                return False

        return True
    except (socket.gaierror, socket.herror, ValueError, OSError):
        # Fail closed on DNS resolution failures or timeouts
        return False


class SafeWebSearchTool(RuntimeTool):
    """Runtime tool for executing safe web searches and external intel retrieval with SSRF defense.

    Attributes:
        timeout: Maximum duration allowed for outbound requests in seconds.
        max_bytes: Maximum payload size allowed for response bodies.
    """

    def __init__(
        self,
        timeout: float = 10.0,
        max_bytes: int = 1024 * 1024,
        api_key: Optional[str] = None,
    ) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.api_key = api_key.strip() if api_key else ""

        super().__init__(
            name="safe_web_search",
            description=(
                "Search the web for company, engineering culture, or market information with SSRF protection."
            ),
            func=self.search,
            param_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or company name to search for",
                    },
                    "target_url": {
                        "type": "string",
                        "description": "Optional direct HTTPS URL to fetch for deep recon",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return",
                    },
                },
                "required": ["query"],
            },
        )

    async def search(
        self,
        query: str,
        target_url: Optional[str] = None,
        max_results: int = 5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute safe search query or fetch verified target URL.

        Args:
            query: Search query string.
            target_url: Optional explicit URL to fetch.
            max_results: Max items to return.

        Returns:
            Dict[str, Any]: Search results or blocked security notification.
        """
        if target_url:
            clean_url = sanitize_secrets(target_url)
            if not await asyncio.to_thread(validate_outbound_url, target_url):
                logger.warning("SSRF blocked attempt to access unsafe URL: %s", clean_url)
                return {
                    "status": "blocked",
                    "error": f"URL '{clean_url}' blocked by SSRF defense policy",
                }

            try:
                import httpx

                async with httpx.AsyncClient(timeout=min(self.timeout, 5.0), follow_redirects=False) as client:
                    async with client.stream("GET", target_url, headers={"User-Agent": "DPL-Pvan-Research/1.0"}) as resp:
                        resp.raise_for_status()
                        content_type = resp.headers.get("content-type", "").casefold()
                        if "text/html" not in content_type and "text/plain" not in content_type:
                            return {"status": "unavailable", "url": clean_url, "reason": "unsupported_content_type"}
                        body = bytearray()
                        async for chunk in resp.aiter_bytes():
                            remaining = self.max_bytes - len(body)
                            if remaining <= 0:
                                break
                            body.extend(chunk[:remaining])
                            if len(chunk) > remaining:
                                break
                        text_content = _readable_page_text(bytes(body), content_type)
                    if not text_content:
                        return {"status": "unavailable", "url": clean_url, "reason": "no_readable_page_text"}
                    return {
                        "status": "success",
                        "url": clean_url,
                        "status_code": resp.status_code,
                        "content": text_content,
                        "content_hash": hashlib.sha256(body).hexdigest(),
                        "content_type": content_type,
                        "truncated": len(body) >= self.max_bytes,
                        "total_bytes_read": len(body),
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    }
            except Exception as req_err:
                clean_err = sanitize_secrets(str(req_err))
                logger.warning("Outbound request to %s failed: %s", clean_url, clean_err)
                return {
                    "status": "error",
                    "url": clean_url,
                    "error": clean_err,
                }

        if not self.api_key:
            return {"status": "unavailable", "reason": "search_provider_not_configured", "query": query}

        try:
            import httpx

            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
                    params={
                        "q": str(query)[:600],
                        "count": max(1, min(int(max_results), 4)),
                        "country": "ALL",
                        "search_lang": "en",
                    },
                )
                response.raise_for_status()
                payload = response.json()
            collected_at = datetime.now(timezone.utc).isoformat()
            results = []
            seen = set()
            for position, item in enumerate((payload.get("web") or {}).get("results") or []):
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                parsed = urllib.parse.urlparse(url)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    continue
                title = str(item.get("title") or "").strip()[:400]
                snippet = str(item.get("description") or "").strip()[:1200]
                canonical = urllib.parse.urlunparse(
                    (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", "", "")
                )
                content_hash = hashlib.sha256(f"{canonical}\n{title}\n{snippet}".encode("utf-8")).hexdigest()
                if canonical in seen:
                    continue
                seen.add(canonical)
                results.append(
                    {
                        "source_id": f"web_{content_hash[:20]}",
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                        "source": url,
                        "source_type": "search_result",
                        "trust_tier": "unreviewed",
                        "verification_status": "unreviewed",
                        "query": str(query)[:600],
                        "position": position + 1,
                        "content_hash": content_hash,
                        "published_at": item.get("published_at"),
                        "source_age": item.get("page_age") or item.get("age"),
                        "collected_at": collected_at,
                    }
                )
                if len(results) >= max(1, min(int(max_results), 4)):
                    break
            return {"status": "success", "query": query, "results": results, "collected_at": collected_at}
        except Exception as req_err:
            clean_err = sanitize_secrets(str(req_err))
            logger.warning("Web search provider request failed: %s", clean_err)
            return {"status": "error", "reason": clean_err, "query": query}


class MarketIntelTool(RuntimeTool):
    """Runtime tool providing curated engineering culture, hiring bar, and tech stack intel."""

    # Curated knowledge base for known tech companies
    CURATED_INTEL: Dict[str, Dict[str, Any]] = {
        "google": {
            "culture": "High engineering bar, scale-first thinking, algorithmic rigor, blameless postmortems.",
            "tech_stack": ["C++", "Java", "Go", "Python", "Kubernetes", "Spanner", "Borg"],
            "hiring_bar": "Deep understanding of algorithms, distributed systems, and Googleyness.",
            "interview_focus": "Distributed systems, scale, data structures, concurrency, resilience.",
        },
        "amazon": {
            "culture": "Leadership Principles, customer obsession, operational excellence, ownership.",
            "tech_stack": ["Java", "Python", "AWS DynamoDB", "AWS Lambda", "S3", "Kafka"],
            "hiring_bar": "Strong alignment with Amazon Leadership Principles and practical system design.",
            "interview_focus": "System design tradeoffs, scalability, incident handling, LP behavioral probes.",
        },
        "techcorp": {
            "culture": "High Bar, pragmatic engineering, strong testing discipline, distributed systems focus.",
            "tech_stack": ["Python", "Go", "Kafka", "PostgreSQL", "Redis", "Docker"],
            "hiring_bar": "Senior level craftsmanship, concurrency mastery, architectural clarity.",
            "interview_focus": "Concurrency, event streaming, data modeling, clean code architecture.",
        },
    }

    def __init__(self) -> None:
        super().__init__(
            name="market_intel",
            description=(
                "Retrieve curated company intelligence including engineering culture, tech stack, and hiring bar."
            ),
            func=self.get_market_intel,
            param_schema={
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Name of the target company to analyze",
                    },
                    "target_role": {
                        "type": "string",
                        "description": "Target job title or level",
                    },
                },
                "required": ["company_name"],
            },
        )

    async def get_market_intel(
        self,
        company_name: str,
        target_role: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Retrieve market intelligence for the specified company.

        Args:
            company_name: Target company name.
            target_role: Target role or seniority.

        Returns:
            Dict[str, Any]: Structured intelligence payload.
        """
        clean_name = company_name.strip().lower()
        if clean_name in self.CURATED_INTEL:
            data = dict(self.CURATED_INTEL[clean_name])
            data["company"] = company_name
            data["target_role"] = target_role or "Software Engineer"
            data["source_type"] = "static_curated_entry"
            data["verification_status"] = "unreviewed"
            return data

        return {
            "company": company_name,
            "target_role": target_role or "Software Engineer",
            "status": "unavailable",
            "reason": "No reviewed company-specific source is available; do not infer company facts.",
        }
