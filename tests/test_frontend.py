"""
tests/test_frontend.py
======================
Phase-11 frontend tests (Node-free).

Tests:
  (a) Frontend file existence + cross-references in index.html
  (b) nginx.conf proxy-rule basics
  (c) docker-compose config validation (via `docker compose config`)
"""

import re
import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[1]
FRONTEND    = REPO_ROOT / "frontend"
INDEX_HTML  = FRONTEND / "index.html"
STYLE_CSS   = FRONTEND / "style.css"
APP_JS      = FRONTEND / "app.js"
NGINX_CONF  = FRONTEND / "nginx.conf"
COMPOSE_YML = REPO_ROOT / "docker-compose.yml"


# ===========================================================================
# (a) File existence + HTML cross-references
# ===========================================================================

class TestFrontendFiles:
    """All three frontend files must exist."""

    def test_index_html_exists(self):
        assert INDEX_HTML.exists(), "frontend/index.html not found"

    def test_style_css_exists(self):
        assert STYLE_CSS.exists(), "frontend/style.css not found"

    def test_app_js_exists(self):
        assert APP_JS.exists(), "frontend/app.js not found"

    def test_nginx_conf_exists(self):
        assert NGINX_CONF.exists(), "frontend/nginx.conf not found"

    def test_index_references_style_css(self):
        """index.html must contain a <link> to style.css."""
        content = INDEX_HTML.read_text(encoding="utf-8")
        assert "style.css" in content, "index.html does not reference style.css"

    def test_index_references_app_js(self):
        """index.html must contain a <script> tag for app.js."""
        content = INDEX_HTML.read_text(encoding="utf-8")
        assert "app.js" in content, "index.html does not reference app.js"

    def test_index_has_single_h1(self):
        """Good HTML practice: exactly one <h1> per page."""
        content = INDEX_HTML.read_text(encoding="utf-8")
        h1_tags = re.findall(r"<h1[\s>]", content, re.IGNORECASE)
        assert len(h1_tags) == 1, (
            f"index.html should have exactly 1 <h1>; found {len(h1_tags)}"
        )

    def test_index_has_lang_attribute(self):
        """Accessibility: <html> must declare lang."""
        content = INDEX_HTML.read_text(encoding="utf-8")
        assert re.search(r"<html[^>]+lang=", content, re.IGNORECASE), (
            "index.html missing lang= attribute on <html>"
        )

    def test_index_has_meta_description(self):
        """SEO: meta description must be present."""
        content = INDEX_HTML.read_text(encoding="utf-8")
        assert re.search(r'<meta[^>]+name=["\']description["\']', content, re.IGNORECASE), (
            "index.html missing <meta name='description'>"
        )

    def test_index_has_viewport_meta(self):
        """Responsive: viewport meta tag required."""
        content = INDEX_HTML.read_text(encoding="utf-8")
        assert re.search(r'<meta[^>]+name=["\']viewport["\']', content, re.IGNORECASE), (
            "index.html missing <meta name='viewport'>"
        )

    def test_app_js_no_innerhtml_with_api_data(self):
        """
        Security: app.js must not assign raw API strings to innerHTML.
        Allowed: clearing innerHTML with a bare empty string literal.
        Disallowed: innerHTML = <variable / expression from API data>.
        """
        content = APP_JS.read_text(encoding="utf-8")
        # Collect everything on the right-hand side of each .innerHTML = expression
        all_assignments = re.findall(r"\.innerHTML\s*=\s*([^\n;]+)", content)
        dangerous = []
        for rhs in all_assignments:
            stripped = rhs.strip().rstrip(";").strip()
            # Allow only bare empty-string literals used to clear a container
            if stripped not in ('""', "''", "``"):
                dangerous.append(stripped)
        assert len(dangerous) == 0, (
            f"app.js has {len(dangerous)} potentially unsafe innerHTML assignment(s): "
            f"{dangerous[:3]}"
        )

    def test_app_js_uses_api_base_path(self):
        """app.js must target /api (nginx proxy path), not a hard-coded port."""
        content = APP_JS.read_text(encoding="utf-8")
        assert '"/api"' in content or "'/api'" in content, (
            "app.js does not define API_BASE = '/api' -- may bypass nginx proxy"
        )

    def test_app_js_defines_low_confidence_threshold(self):
        """Phase 11b: app.js must define LOW_CONFIDENCE_THRESHOLD constant (default 0.40)."""
        content = APP_JS.read_text(encoding="utf-8")
        match = re.search(r"const\s+LOW_CONFIDENCE_THRESHOLD\s*=\s*([0-9.]+)", content)
        assert match, "LOW_CONFIDENCE_THRESHOLD constant not found in app.js"
        assert float(match.group(1)) == 0.40, (
            f"Expected LOW_CONFIDENCE_THRESHOLD 0.40, got {match.group(1)}"
        )

    def test_app_js_defines_symptom_display_map(self):
        """Phase 11b: app.js must define SYMPTOM_DISPLAY_MAP with typo fixes."""
        content = APP_JS.read_text(encoding="utf-8")
        assert "SYMPTOM_DISPLAY_MAP" in content, (
            "SYMPTOM_DISPLAY_MAP not found in app.js"
        )
        assert "cold_hands_and_feets" in content and "cold hands and feet" in content, (
            "cold_hands_and_feets mapping missing in SYMPTOM_DISPLAY_MAP"
        )
        assert "spotting_urination" in content and "spotting while urinating" in content, (
            "spotting_urination mapping missing in SYMPTOM_DISPLAY_MAP"
        )

    def test_index_has_low_confidence_notice(self):
        """Phase 11b: index.html must have low-confidence-notice element."""
        content = INDEX_HTML.read_text(encoding="utf-8")
        assert 'id="low-confidence-notice"' in content or "id='low-confidence-notice'" in content, (
            "low-confidence-notice element missing in index.html"
        )


# ===========================================================================
# (b) nginx.conf proxy rule
# ===========================================================================

class TestNginxConf:
    """nginx.conf must contain the /api/ proxy block pointing to api:8000."""

    def _conf(self):
        return NGINX_CONF.read_text(encoding="utf-8")

    def test_listens_on_port_80(self):
        assert re.search(r"listen\s+80\b", self._conf()), (
            "nginx.conf does not listen on port 80"
        )

    def test_proxy_pass_to_api(self):
        """proxy_pass must forward to http://api:8000."""
        assert "proxy_pass" in self._conf(), "nginx.conf missing proxy_pass directive"
        assert "api:8000" in self._conf(), (
            "nginx.conf proxy_pass does not target api:8000"
        )

    def test_api_location_block(self):
        """Must have a location block for /api/."""
        assert re.search(r"location\s+/api/", self._conf()), (
            "nginx.conf missing 'location /api/' block"
        )

    def test_proxy_strips_api_prefix(self):
        """
        The rewrite rule (or proxy_pass with trailing slash) must strip /api prefix
        so FastAPI sees /health not /api/health.
        """
        conf = self._conf()
        has_rewrite = bool(re.search(r"rewrite\s+\^/api/", conf))
        has_trailing_slash = bool(re.search(r"proxy_pass\s+http://api:8000/\s*;", conf))
        assert has_rewrite or has_trailing_slash, (
            "nginx.conf must strip the /api prefix (rewrite or trailing-slash proxy_pass)"
        )

    def test_root_location_serves_index(self):
        """Root location block must exist to serve the SPA."""
        assert re.search(r"location\s+/\s*\{", self._conf()), (
            "nginx.conf missing root location block"
        )


# ===========================================================================
# (c) docker-compose config validation
# ===========================================================================

class TestDockerComposeConfig:
    """docker compose config must parse cleanly and include the frontend service."""

    @pytest.fixture(scope="class")
    @classmethod
    def compose_config(cls):
        """Run `docker compose config` and return the completed process."""
        result = subprocess.run(
            ["docker", "compose", "config"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        return result

    def test_compose_config_exits_zero(self, compose_config):
        assert compose_config.returncode == 0, (
            f"docker compose config failed:\n{compose_config.stderr}"
        )

    def test_compose_config_includes_frontend_service(self, compose_config):
        assert "disease-frontend" in compose_config.stdout or \
               "frontend" in compose_config.stdout, (
            "docker compose config output does not mention the frontend service"
        )

    def test_compose_config_includes_api_service(self, compose_config):
        assert "disease-api" in compose_config.stdout or \
               "  api:" in compose_config.stdout, (
            "docker compose config output does not mention the api service"
        )

    def test_compose_frontend_port_8080(self, compose_config):
        """Frontend must be mapped to host port 8080."""
        assert "8080" in compose_config.stdout, (
            "docker compose config does not show port 8080 for frontend"
        )
