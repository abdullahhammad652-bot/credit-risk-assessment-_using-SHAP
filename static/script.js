// Leave empty when the page is served by the same FastAPI app (recommended).
// Set to e.g. "https://your-api.onrender.com" if the frontend is hosted separately.
const API_URL = "";

const form = document.getElementById("form");
const submitBtn = document.getElementById("submit");
const errorBox = document.getElementById("error");
const pct = document.getElementById("pct");
const result = document.getElementById("result");
const fill = document.getElementById("fill");
const needle = document.getElementById("needle");
const tick = document.getElementById("tick");
const probEl = document.getElementById("prob");
const verdictEl = document.getElementById("verdict");
const explainEl = document.getElementById("explain");
const facts = document.getElementById("facts");

const PRESETS = {
  low: {
    person_age: 42, person_income: 120000, person_home_ownership: "MORTGAGE",
    person_emp_length: 14, loan_intent: "HOMEIMPROVEMENT", loan_grade: "A",
    loan_amnt: 9000, loan_int_rate: 6.8, cb_person_default_on_file: "N",
    cb_person_cred_hist_length: 18,
  },
  high: {
    person_age: 24, person_income: 18000, person_home_ownership: "RENT",
    person_emp_length: 0, loan_intent: "PERSONAL", loan_grade: "F",
    loan_amnt: 12000, loan_int_rate: 21.5, cb_person_default_on_file: "Y",
    cb_person_cred_hist_length: 2,
  },
};

/* ---------- loan share of income (auto) ---------- */
function loanShare() {
  const amnt = parseFloat(form.loan_amnt.value);
  const income = parseFloat(form.person_income.value);
  return amnt > 0 && income > 0 ? amnt / income : null;
}

function updateShare() {
  const v = loanShare();
  pct.textContent = v === null ? "Calculated from amount and income" : `${(v * 100).toFixed(1)}% of annual income`;
  pct.classList.toggle("has-value", v !== null);
}
form.addEventListener("input", updateShare);

/* ---------- presets ---------- */
document.querySelectorAll("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const key = btn.dataset.preset;
    if (key === "clear") {
      form.reset();
    } else {
      Object.entries(PRESETS[key]).forEach(([name, value]) => (form.elements[name].value = value));
    }
    form.querySelectorAll(".invalid").forEach((el) => el.classList.remove("invalid"));
    hideError();
    updateShare();
    resetResult();
  });
});

/* ---------- helpers ---------- */
function showError(msg) { errorBox.textContent = msg; errorBox.hidden = false; }
function hideError() { errorBox.hidden = true; }

function setLoading(on) {
  submitBtn.disabled = on;
  submitBtn.classList.toggle("loading", on);
  submitBtn.querySelector(".label").textContent = on ? "Assessing" : "Assess risk";
}

function countUp(el, to, ms = 1400) {
  const start = performance.now();
  const ease = (t) => 1 - Math.pow(1 - t, 4);
  (function frame(now) {
    const t = Math.min((now - start) / ms, 1);
    el.textContent = (to * ease(t)).toFixed(1);
    if (t < 1) requestAnimationFrame(frame);
  })(start);
}

function placeTick(threshold) {
  const a = Math.PI * threshold;
  const cx = 120, cy = 120;
  const x1 = cx - 88 * Math.cos(a), y1 = cy - 88 * Math.sin(a);
  const x2 = cx - 110 * Math.cos(a), y2 = cy - 110 * Math.sin(a);
  const tx = cx - 124 * Math.cos(a), ty = cy - 124 * Math.sin(a);
  const line = tick.querySelector("line");
  line.setAttribute("x1", x1); line.setAttribute("y1", y1);
  line.setAttribute("x2", x2); line.setAttribute("y2", y2);
  const text = tick.querySelector("text");
  text.setAttribute("x", tx); text.setAttribute("y", ty + 3);
  tick.setAttribute("opacity", 1);
}

function resetResult() {
  result.dataset.state = "idle";
  result.classList.remove("pop");
  fill.style.strokeDashoffset = 100;
  needle.style.transform = "rotate(0deg)";
  tick.setAttribute("opacity", 0);
  probEl.textContent = "--";
  verdictEl.textContent = "Waiting for details";
  explainEl.textContent = "Fill in the form or load an example, then select Assess risk.";
  facts.hidden = true;
}

/* ---------- validation ---------- */
function validate() {
  let ok = true;
  form.querySelectorAll("input, select").forEach((el) => {
    const bad = !el.checkValidity();
    el.classList.toggle("invalid", bad);
    if (bad) ok = false;
  });
  return ok;
}

/* ---------- submit ---------- */
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();

  if (!validate()) {
    showError("Some fields are empty or out of range. Check the highlighted fields and try again.");
    return;
  }

  const share = loanShare();
  const payload = {
    person_age: parseInt(form.person_age.value, 10),
    person_income: parseFloat(form.person_income.value),
    person_home_ownership: form.person_home_ownership.value,
    person_emp_length: parseInt(form.person_emp_length.value, 10),
    loan_intent: form.loan_intent.value,
    loan_grade: form.loan_grade.value,
    loan_amnt: parseFloat(form.loan_amnt.value),
    loan_int_rate: parseFloat(form.loan_int_rate.value),
    loan_percent_income: Math.round(share * 10000) / 10000,
    cb_person_default_on_file: form.cb_person_default_on_file.value,
    cb_person_cred_hist_length: parseInt(form.cb_person_cred_hist_length.value, 10),
  };

  setLoading(true);
  try {
    const res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const detail = Array.isArray(body.detail)
        ? body.detail.map((d) => `${d.loc.slice(-1)}: ${d.msg}`).join("; ")
        : body.detail || `Server returned ${res.status}`;
      throw new Error(detail);
    }
    render(await res.json(), share);
  } catch (err) {
    showError(`Could not get a result. ${err.message}. If the service was idle, wait a few seconds and try again.`);
  } finally {
    setLoading(false);
  }
});

/* ---------- render result ---------- */
function render(data, share) {
  const p = data.default_probabilty;
  const t = data.Threshold;
  const high = data.default_prediction === 1;

  // reset so the sweep animates from zero each time
  fill.style.transition = needle.style.transition = "none";
  fill.style.strokeDashoffset = 100;
  needle.style.transform = "rotate(0deg)";
  result.classList.remove("pop");
  void result.offsetWidth;
  fill.style.transition = needle.style.transition = "";

  result.dataset.state = high ? "high" : "low";
  placeTick(t);

  requestAnimationFrame(() => {
    fill.style.strokeDashoffset = 100 - p * 100;
    needle.style.transform = `rotate(${p * 180}deg)`;
    countUp(probEl, p * 100);
    result.classList.add("pop");
  });

  verdictEl.textContent = high ? "High risk" : "Low risk";
  explainEl.textContent = high
    ? "The default chance is at or above the model's cutoff. Review this application closely."
    : "The default chance is below the model's cutoff.";

  document.getElementById("f-threshold").textContent = `${(t * 100).toFixed(1)}%`;
  document.getElementById("f-lti").textContent = `${(share * 100).toFixed(1)}%`;
  facts.hidden = false;

  if (window.matchMedia("(max-width: 900px)").matches) {
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}
