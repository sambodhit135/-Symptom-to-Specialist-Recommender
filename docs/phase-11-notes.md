# Phase 11 — Frontend Web Interface

> **Branch:** `feature/phase-11-frontend`
> **Goal:** A single-page vanilla HTML/CSS/JS UI served by nginx, proxying API calls to the existing FastAPI backend.

---

## Architecture: Browser → nginx → FastAPI → Model

```
User's Browser
      │
      │  http://localhost:8080
      ▼
┌─────────────────────────────────────────────┐
│  nginx:1.27-alpine  (container: disease-    │
│  frontend, port 80 inside / 8080 on host)   │
│                                             │
│  GET /          → serve index.html          │
│  GET /style.css → serve style.css           │
│  GET /app.js    → serve app.js              │
│                                             │
│  GET  /api/*    ─────────────────────────── │
│  POST /api/*    → strip /api prefix         │
│                 → proxy_pass api:8000       │
└──────────────────┬──────────────────────────┘
                   │  Docker internal network (app-net)
                   │  http://api:8000
                   ▼
┌─────────────────────────────────────────────┐
│  FastAPI (uvicorn)  container: disease-api  │
│                                             │
│  GET  /health        ← /api/health          │
│  GET  /symptoms      ← /api/symptoms        │
│  GET  /model-info    ← /api/model-info      │
│  POST /predict       ← /api/predict         │
│  POST /feedback      ← /api/feedback        │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
          Random-Forest model
          models/production/model.joblib
```

**Why same-origin via nginx?**
The browser talks to nginx on `localhost:8080`. All API calls use the path `/api/…`, which is on the *same origin*, so no CORS preflight is needed and we do not need the existing `allow_origins=["*"]` wildcard on the FastAPI side.

---

## Files Created

| File | Purpose |
|------|---------|
| `frontend/index.html` | Single HTML page: status bar, symptom selector, results, emergency banner, feedback |
| `frontend/style.css` | Dark-mode design system — CSS custom properties, no external deps |
| `frontend/app.js` | Vanilla ES2020 JS — fetches API, renders DOM via `createElement`/`textContent` (XSS-safe) |
| `frontend/nginx.conf` | Static file server + `/api/` reverse proxy with prefix stripping |
| `tests/test_frontend.py` | pytest tests (file existence, HTML structure, nginx conf, docker compose config) |

---

## User Interaction Flow (Step by Step)

### 1 · Page load
1. `DOMContentLoaded` fires → `init()` in `app.js` runs.
2. `refreshStatusBar()` calls `GET /api/health` and updates the status pills.
3. `loadSymptoms()` calls `GET /api/symptoms` → renders 131 symptom chips in a scrollable grid.

### 2 · Symptom selection
- User types in the search box → `onSearchInput()` filters chips client-side (no extra network request).
- Clicking a chip (or its checkbox) calls `onSymptomToggle()` → updates `state.selectedSet` + re-renders removable tag pills.
- Clicking the `×` on a tag calls `removeSymptom()` → unchecks the chip and removes the tag.

### 3 · Prediction
1. User clicks **"Find Specialist"** → `onPredictClick()`.
2. Validates at least one symptom is selected; shows inline error if not.
3. Sets button to loading state (spinner visible, text hidden, disabled).
4. Calls `POST /api/predict` with `{ "symptoms": [...] }`.
5. On success:
   - If `emergency_flag === true` → shows animated red banner with `emergency_reasons`.
   - Renders three result cards (top result highlighted in blue): disease, confidence bar, specialist.
   - Shows the mandatory disclaimer.
   - Shows the feedback section.
6. On failure (network down, API error) → shows a readable error box, never a blank page.

### 4 · Feedback
- **"Yes"** → `submitFeedback(true, null)` → `POST /api/feedback`.
- **"No"** → reveals a disease-correction dropdown (populated lazily from `GET /api/diseases`).
  - User optionally picks the correct disease → clicks **"Submit correction"** → `POST /api/feedback` with `helpful: false, true_disease: <name>`.
- Thank-you message shown after submission; buttons disabled to prevent double-submit.

---

## nginx Proxy — How Prefix Stripping Works

```nginx
location /api/ {
    rewrite ^/api/(.*)$ /$1 break;  # strip /api prefix
    proxy_pass http://api:8000;
}
```

| Browser request | nginx forwards to FastAPI as |
|-----------------|------------------------------|
| `GET /api/health` | `GET http://api:8000/health` |
| `POST /api/predict` | `POST http://api:8000/predict` |
| `POST /api/feedback` | `POST http://api:8000/feedback` |

---

## Security Notes

| Concern | Mitigation |
|---------|-----------|
| XSS | All API text inserted via `textContent` / `createElement` — never `innerHTML` with dynamic data |
| Clickjacking | `X-Frame-Options: SAMEORIGIN` header in nginx |
| MIME sniffing | `X-Content-Type-Options: nosniff` header in nginx |
| CORS | Not needed — browser and API share origin via nginx proxy |
| Admin endpoint | `/admin/reload-model` requires `X-API-Key` header — not exposed in the UI |

---

## Running the Frontend

```powershell
# From repo root (Docker Desktop must be running)
docker compose up -d --build

# Frontend available at:
Start-Process "http://localhost:8080"

# Verify via proxy:
Invoke-RestMethod http://localhost:8080/api/health

# Stop everything:
docker compose down
```

---

## Screenshots to Take for Presentation

1. **Symptom selection** — chips loaded, a few selected, tags visible below.
2. **Search filter active** — search box has text, chip list filtered.
3. **Normal result** — 3 result cards visible, confidence bars animated, specialist shown.
4. **Emergency result** — red pulsing banner visible alongside result cards (use symptoms: `chest_pain`, `breathlessness`, `sweating`).
5. **Feedback submitted** — "Thank you" message showing below the feedback buttons.
6. **API offline error** — stop `disease-api` container; reload page → friendly error message in the symptom panel and predict error box.
7. **Status bar: API offline** — red dot visible on status bar after stopping the api container.
