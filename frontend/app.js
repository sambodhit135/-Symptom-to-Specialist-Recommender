/**
 * frontend/app.js
 * ===========================================================================
 * Disease Prediction UI — vanilla JavaScript (ES2020, no frameworks).
 *
 * Flow:
 *   1. On page load, fetch symptom list from GET /api/symptoms and render chips.
 *   2. User searches / selects symptoms → selected set is maintained in memory.
 *   3. "Find Specialist" calls POST /api/predict → renders result cards,
 *      emergency banner, and feedback section.
 *   4. Feedback "Yes"/"No" submits POST /api/feedback.
 *   5. Status bar polls GET /api/health + GET /api/model-info once on load.
 *
 * Security note: All text from the API is inserted via textContent / DOM APIs,
 * never via innerHTML, to prevent XSS.
 * ===========================================================================
 */

"use strict";

/* ---------------------------------------------------------------------------
 * CONSTANTS — base URL is empty string because nginx proxies /api/* → API
 * --------------------------------------------------------------------------- */
const API_BASE = "/api";

/* ---------------------------------------------------------------------------
 * STATE
 * --------------------------------------------------------------------------- */
const state = {
  allSymptoms:    [],      // full list from /api/symptoms
  selectedSet:    new Set(), // currently selected symptom names
  lastRequestId:  null,    // request_id from last /api/predict response
  allDiseases:    [],      // list from /api/diseases (populated on demand)
  feedbackDone:   false,   // prevent double-submit
};

/* ---------------------------------------------------------------------------
 * DOM REFERENCES — collected once at startup
 * --------------------------------------------------------------------------- */
const dom = {
  // Status bar
  statusBar:          document.getElementById("status-bar"),
  dotApi:             document.getElementById("dot-api"),
  labelApi:           document.getElementById("label-api"),
  dotModel:           document.getElementById("dot-model"),
  labelModel:         document.getElementById("label-model"),

  // Symptom panel
  symptomSearch:      document.getElementById("symptom-search"),
  symptomList:        document.getElementById("symptom-list"),
  selectedTags:       document.getElementById("selected-tags"),
  noTagsHint:         document.getElementById("no-tags-hint"),

  // Predict button
  predictBtn:         document.getElementById("predict-btn"),
  inputError:         document.getElementById("input-error"),

  // Emergency banner
  emergencyBanner:    document.getElementById("emergency-banner"),
  emergencyReasons:   document.getElementById("emergency-reasons"),

  // Results
  resultsSection:     document.getElementById("results-section"),
  resultCards:        document.getElementById("result-cards"),
  resultDisclaimer:   document.getElementById("result-disclaimer"),
  apiErrorBox:        document.getElementById("api-error-box"),

  // Feedback
  feedbackSection:    document.getElementById("feedback-section"),
  fbYes:              document.getElementById("fb-yes"),
  fbNo:               document.getElementById("fb-no"),
  correctionRow:      document.getElementById("correction-row"),
  trueDiseaseSelect:  document.getElementById("true-disease-select"),
  submitCorrectionBtn:document.getElementById("submit-correction-btn"),
  feedbackThanks:     document.getElementById("feedback-thanks"),
};

/* ---------------------------------------------------------------------------
 * UTILITY: safe text helper — creates a text node (no XSS risk)
 * --------------------------------------------------------------------------- */
function txt(str) {
  return document.createTextNode(String(str));
}

/* ---------------------------------------------------------------------------
 * UTILITY: show / hide helper (uses a CSS class "visible")
 * --------------------------------------------------------------------------- */
function show(el) { el.classList.add("visible"); }
function hide(el) { el.classList.remove("visible"); }
function setVisible(el, visible) { visible ? show(el) : hide(el); }

/* ---------------------------------------------------------------------------
 * UTILITY: display name — replaces underscores with spaces for the UI
 * --------------------------------------------------------------------------- */
function displayName(symptomName) {
  return symptomName.replace(/_/g, " ");
}

/* ---------------------------------------------------------------------------
 * FETCH WRAPPER
 * Makes a JSON fetch and returns { data, ok, error }.
 * Never throws — errors are returned as { ok: false, error: <string> }.
 * --------------------------------------------------------------------------- */
async function apiFetch(path, options = {}) {
  try {
    const response = await fetch(API_BASE + path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const msg = data?.detail || `HTTP ${response.status}`;
      return { ok: false, error: msg, status: response.status };
    }
    return { ok: true, data };
  } catch (err) {
    // Network error or backend completely down
    return {
      ok: false,
      error: "Cannot reach the API. Check that the backend is running.",
    };
  }
}

/* ---------------------------------------------------------------------------
 * STATUS BAR — fetches /api/health and /api/model-info once on load
 * --------------------------------------------------------------------------- */
async function refreshStatusBar() {
  const { ok, data } = await apiFetch("/health");

  if (ok && data.status === "healthy") {
    dom.dotApi.className   = "status-dot ok";
    dom.labelApi.textContent = "API online";
    dom.dotModel.className = "status-dot ok";
    dom.labelModel.textContent = `Model v${data.model_version}`;
  } else {
    dom.dotApi.className   = "status-dot err";
    dom.labelApi.textContent = "API offline";
    dom.dotModel.className = "status-dot err";
    dom.labelModel.textContent = "Model unknown";
  }
}

/* ---------------------------------------------------------------------------
 * LOAD SYMPTOMS — GET /api/symptoms
 * Populates state.allSymptoms and renders the chip list.
 * --------------------------------------------------------------------------- */
async function loadSymptoms() {
  const { ok, data, error } = await apiFetch("/symptoms");

  if (!ok) {
    // Show error inside symptom panel
    const errEl = document.createElement("p");
    errEl.className = "message-box error";
    errEl.appendChild(txt(`Failed to load symptoms: ${error}`));
    dom.symptomList.appendChild(errEl);
    return;
  }

  state.allSymptoms = data.symptoms || [];
  renderSymptomList(state.allSymptoms);
}

/* ---------------------------------------------------------------------------
 * RENDER SYMPTOM LIST — builds a chip (label+checkbox) for each symptom
 * --------------------------------------------------------------------------- */
function renderSymptomList(symptoms) {
  // Clear existing content
  dom.symptomList.innerHTML = "";

  if (symptoms.length === 0) {
    const p = document.createElement("p");
    p.className = "no-tags-hint";
    p.appendChild(txt("No symptoms match your search."));
    dom.symptomList.appendChild(p);
    return;
  }

  symptoms.forEach((name) => {
    const label = document.createElement("label");
    label.className = "symptom-chip" + (state.selectedSet.has(name) ? " selected" : "");
    label.setAttribute("for", `chk-${name}`);

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id   = `chk-${name}`;
    checkbox.name = name;
    checkbox.value = name;
    checkbox.checked = state.selectedSet.has(name);
    // ARIA: associate label text
    checkbox.setAttribute("aria-label", displayName(name));

    const span = document.createElement("span");
    span.appendChild(txt(displayName(name)));

    checkbox.addEventListener("change", () => onSymptomToggle(name, checkbox.checked));

    label.appendChild(checkbox);
    label.appendChild(span);
    dom.symptomList.appendChild(label);
  });
}

/* ---------------------------------------------------------------------------
 * SYMPTOM TOGGLE — called whenever a chip checkbox changes
 * --------------------------------------------------------------------------- */
function onSymptomToggle(name, checked) {
  if (checked) {
    state.selectedSet.add(name);
  } else {
    state.selectedSet.delete(name);
  }
  syncChipHighlight(name, checked);
  renderSelectedTags();
  clearResults(); // clear old results when selection changes
}

/* ---------------------------------------------------------------------------
 * SYNC CHIP HIGHLIGHT — add/remove .selected class on the chip label
 * --------------------------------------------------------------------------- */
function syncChipHighlight(name, selected) {
  const chk = document.getElementById(`chk-${name}`);
  if (chk) {
    const label = chk.closest(".symptom-chip");
    if (label) label.classList.toggle("selected", selected);
  }
}

/* ---------------------------------------------------------------------------
 * RENDER SELECTED TAGS — shows removable tag pills above the predict button
 * --------------------------------------------------------------------------- */
function renderSelectedTags() {
  dom.selectedTags.innerHTML = "";

  if (state.selectedSet.size === 0) {
    dom.noTagsHint.style.display = "block";
    return;
  }
  dom.noTagsHint.style.display = "none";

  state.selectedSet.forEach((name) => {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.appendChild(txt(displayName(name)));

    const btn = document.createElement("button");
    btn.className = "tag-remove";
    btn.setAttribute("aria-label", `Remove ${displayName(name)}`);
    btn.appendChild(txt("×"));
    btn.addEventListener("click", () => removeSymptom(name));

    tag.appendChild(btn);
    dom.selectedTags.appendChild(tag);
  });
}

/* ---------------------------------------------------------------------------
 * REMOVE SYMPTOM — deselects a symptom via the tag ×
 * --------------------------------------------------------------------------- */
function removeSymptom(name) {
  state.selectedSet.delete(name);
  // Uncheck the chip if it is currently rendered
  const chk = document.getElementById(`chk-${name}`);
  if (chk) {
    chk.checked = false;
    const label = chk.closest(".symptom-chip");
    if (label) label.classList.remove("selected");
  }
  renderSelectedTags();
  clearResults();
}

/* ---------------------------------------------------------------------------
 * SEARCH FILTER — filters the chip list as the user types
 * --------------------------------------------------------------------------- */
function onSearchInput() {
  const query = dom.symptomSearch.value.trim().toLowerCase();
  if (query === "") {
    renderSymptomList(state.allSymptoms);
  } else {
    const filtered = state.allSymptoms.filter((s) =>
      s.toLowerCase().includes(query) || displayName(s).toLowerCase().includes(query)
    );
    renderSymptomList(filtered);
  }
}

/* ---------------------------------------------------------------------------
 * CLEAR RESULTS — hides results, emergency banner, and feedback panel
 * --------------------------------------------------------------------------- */
function clearResults() {
  hide(dom.resultsSection);
  hide(dom.emergencyBanner);
  hide(dom.feedbackSection);
  dom.inputError.textContent  = "";
  dom.inputError.style.display = "none";
  dom.apiErrorBox.textContent  = "";
  dom.apiErrorBox.style.display = "none";
  state.lastRequestId = null;
  state.feedbackDone  = false;
}

/* ---------------------------------------------------------------------------
 * PREDICT — POST /api/predict
 * --------------------------------------------------------------------------- */
async function onPredictClick() {
  // Clear previous results / errors
  clearResults();

  // Validation: at least one symptom required
  if (state.selectedSet.size === 0) {
    dom.inputError.textContent = "Please select at least one symptom before predicting.";
    dom.inputError.style.display = "block";
    return;
  }

  // Set loading state
  dom.predictBtn.classList.add("loading");
  dom.predictBtn.disabled = true;

  const symptoms = Array.from(state.selectedSet);

  const { ok, data, error } = await apiFetch("/predict", {
    method: "POST",
    body:   JSON.stringify({ symptoms }),
  });

  // Remove loading state
  dom.predictBtn.classList.remove("loading");
  dom.predictBtn.disabled = false;

  if (!ok) {
    dom.apiErrorBox.textContent = `Prediction failed: ${error}`;
    dom.apiErrorBox.style.display = "block";
    return;
  }

  // Store request_id for feedback
  state.lastRequestId = data.request_id;

  // Render emergency banner if needed
  renderEmergencyBanner(data.emergency_flag, data.emergency_reasons);

  // Render result cards
  renderResults(data.predictions, data.disclaimer);

  // Show feedback section (reset state first)
  resetFeedback();
  // Load disease list for correction dropdown (lazy)
  await ensureDiseasesLoaded();
  show(dom.feedbackSection);
}

/* ---------------------------------------------------------------------------
 * RENDER EMERGENCY BANNER
 * --------------------------------------------------------------------------- */
function renderEmergencyBanner(flag, reasons) {
  if (!flag) {
    hide(dom.emergencyBanner);
    return;
  }

  // Clear previous reasons list
  dom.emergencyReasons.innerHTML = "";

  reasons.forEach((reason) => {
    const li = document.createElement("li");
    li.appendChild(txt(reason));
    dom.emergencyReasons.appendChild(li);
  });

  show(dom.emergencyBanner);
}

/* ---------------------------------------------------------------------------
 * RENDER RESULT CARDS — top-3 predictions
 * --------------------------------------------------------------------------- */
function renderResults(predictions, disclaimer) {
  dom.resultCards.innerHTML = "";

  predictions.forEach((pred, idx) => {
    const rank = idx + 1;
    const pct  = Math.round(pred.confidence * 100);

    const card = document.createElement("article");
    card.className = `result-card rank-${rank}`;
    card.setAttribute("aria-label", `Rank ${rank}: ${pred.disease}`);

    // -- Header row: rank label + disease name + confidence %
    const header = document.createElement("div");
    header.className = "result-header";

    const rankLabel = document.createElement("span");
    rankLabel.className = "result-rank";
    rankLabel.appendChild(txt(rank === 1 ? "Most likely" : rank === 2 ? "2nd possibility" : "3rd possibility"));

    const diseaseName = document.createElement("span");
    diseaseName.className = "result-disease";
    diseaseName.appendChild(txt(pred.disease));

    const confPct = document.createElement("span");
    confPct.className = "result-confidence-pct";
    confPct.setAttribute("aria-label", `Confidence: ${pct}%`);
    confPct.appendChild(txt(`${pct}%`));

    header.appendChild(rankLabel);
    header.appendChild(diseaseName);
    header.appendChild(confPct);

    // -- Confidence bar
    const barWrap = document.createElement("div");
    barWrap.className = "confidence-bar-wrap";
    barWrap.setAttribute("role", "progressbar");
    barWrap.setAttribute("aria-valuenow", pct);
    barWrap.setAttribute("aria-valuemin", "0");
    barWrap.setAttribute("aria-valuemax", "100");

    const barFill = document.createElement("div");
    barFill.className = "confidence-bar-fill";
    barFill.style.width = "0%"; // animate in
    barWrap.appendChild(barFill);
    // Animate after paint
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { barFill.style.width = `${pct}%`; });
    });

    // -- Specialist
    const specialistRow = document.createElement("p");
    specialistRow.className = "result-specialist";
    const strong = document.createElement("strong");
    strong.appendChild(txt(pred.specialist));
    specialistRow.appendChild(txt("Consult a "));
    specialistRow.appendChild(strong);

    card.appendChild(header);
    card.appendChild(barWrap);
    card.appendChild(specialistRow);
    dom.resultCards.appendChild(card);
  });

  // Disclaimer
  dom.resultDisclaimer.textContent = disclaimer;

  show(dom.resultsSection);
}

/* ---------------------------------------------------------------------------
 * LOAD DISEASES — lazy, populates the correction dropdown
 * --------------------------------------------------------------------------- */
async function ensureDiseasesLoaded() {
  if (state.allDiseases.length > 0) return; // already loaded

  // Try /api/diseases first; if 404, nothing to populate (select stays empty)
  const { ok, data } = await apiFetch("/diseases");
  if (ok && Array.isArray(data)) {
    state.allDiseases = data;
  } else if (ok && Array.isArray(data?.diseases)) {
    state.allDiseases = data.diseases;
  }

  // Populate the correction <select>
  const select = dom.trueDiseaseSelect;
  select.innerHTML = ""; // clear old options

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.appendChild(txt("— select correct disease —"));
  select.appendChild(placeholder);

  state.allDiseases.forEach((d) => {
    const opt = document.createElement("option");
    opt.value = d;
    opt.appendChild(txt(d));
    select.appendChild(opt);
  });
}

/* ---------------------------------------------------------------------------
 * FEEDBACK
 * --------------------------------------------------------------------------- */

/** Reset feedback UI to initial state */
function resetFeedback() {
  state.feedbackDone = false;
  dom.fbYes.disabled = false;
  dom.fbNo.disabled  = false;
  hide(dom.correctionRow);
  hide(dom.feedbackThanks);
  dom.trueDiseaseSelect.value = "";
}

/** Called when user clicks "Yes" */
async function onFeedbackYes() {
  if (state.feedbackDone) return;
  await submitFeedback(true, null);
}

/** Called when user clicks "No" */
function onFeedbackNo() {
  if (state.feedbackDone) return;
  show(dom.correctionRow);
}

/** Called when user clicks "Submit correction" */
async function onSubmitCorrection() {
  const trueDisease = dom.trueDiseaseSelect.value || null;
  await submitFeedback(false, trueDisease);
}

/** POST /api/feedback */
async function submitFeedback(helpful, trueDisease) {
  if (!state.lastRequestId) return;

  dom.fbYes.disabled = true;
  dom.fbNo.disabled  = true;

  const body = {
    request_id:   state.lastRequestId,
    helpful:      helpful,
    true_disease: trueDisease || undefined,
  };

  const { ok, error } = await apiFetch("/feedback", {
    method: "POST",
    body:   JSON.stringify(body),
  });

  if (ok) {
    state.feedbackDone = true;
    hide(dom.correctionRow);
    dom.feedbackThanks.textContent = helpful
      ? "Thank you for the positive feedback!"
      : "Thank you — your correction has been recorded.";
    show(dom.feedbackThanks);
  } else {
    // Show a small error but don't block the UI
    dom.feedbackThanks.textContent = `Feedback not saved: ${error}`;
    dom.feedbackThanks.style.color = "var(--clr-danger)";
    show(dom.feedbackThanks);
  }
}

/* ---------------------------------------------------------------------------
 * INIT — wire up events and start async data loads
 * --------------------------------------------------------------------------- */
function init() {
  // Status bar
  refreshStatusBar();

  // Load symptom chips
  loadSymptoms();

  // Search input
  dom.symptomSearch.addEventListener("input", onSearchInput);

  // Predict button
  dom.predictBtn.addEventListener("click", onPredictClick);

  // Feedback buttons
  dom.fbYes.addEventListener("click", onFeedbackYes);
  dom.fbNo.addEventListener("click", onFeedbackNo);
  dom.submitCorrectionBtn.addEventListener("click", onSubmitCorrection);

  // Keyboard: allow pressing Enter in search to trigger predict if chips are loaded
  dom.symptomSearch.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      dom.predictBtn.click();
    }
  });
}

// Run after DOM is ready
document.addEventListener("DOMContentLoaded", init);
