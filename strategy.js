document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("strategy-form");
  const statusEl = document.getElementById("strategy-status");
  const submitBtn = document.getElementById("strategy-submit-btn");
  const resultsEl = document.getElementById("strategy-results");

  const feasibilityBadge = document.getElementById("feasibility-badge");
  const feasibilitySummary = document.getElementById("feasibility-summary");
  const feasibilityDetail = document.getElementById("feasibility-detail");
  const riskWarningEl = document.getElementById("risk-warning");
  const growthChartContainer = document.getElementById("growth-chart-container");
  const growthChartImg = document.getElementById("growth-chart-img");

  const strategyTypeEl = document.getElementById("strategy-type");
  const strategyExplanationEl = document.getElementById("strategy-explanation");
  const assetsListEl = document.getElementById("assets-list");
  const examplesListEl = document.getElementById("examples-list");

  function setStatus(text, type) {
    statusEl.textContent = text;
    statusEl.className = "status " + (type || "");
  }

  function formatPercent(decimal) {
    if (decimal === null || decimal === undefined) return "N/A";
    const pct = decimal * 100;
    if (!isFinite(pct)) return "N/A";
    return pct.toFixed(1) + "%";
  }

  function setFeasibilityBadge(label) {
    feasibilityBadge.className = "badge badge-neutral";
    let text = "Feasibility";
    if (!label) {
      feasibilityBadge.textContent = text;
      return;
    }

    const normalized = label.toLowerCase();
    if (normalized === "very_conservative" || normalized === "realistic") {
      feasibilityBadge.className = "badge badge-feasible";
    } else if (normalized === "aggressive") {
      feasibilityBadge.className = "badge badge-aggressive";
    } else if (normalized === "extremely_unlikely") {
      feasibilityBadge.className = "badge badge-unlikely";
    } else {
      feasibilityBadge.className = "badge badge-neutral";
    }

    if (normalized === "already_met") {
      text = "Goal Already Met";
    } else if (normalized === "very_conservative") {
      text = "Very Conservative";
    } else if (normalized === "realistic") {
      text = "Realistic";
    } else if (normalized === "aggressive") {
      text = "Aggressive";
    } else if (normalized === "extremely_unlikely") {
      text = "Extremely Unlikely";
    } else {
      text = "Feasibility";
    }
    feasibilityBadge.textContent = text;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const goal = (document.getElementById("goal-input").value || "").trim();
    const currentStr = document.getElementById("current-amount-input").value;
    const targetStr = document.getElementById("target-amount-input").value;
    const horizonStr = document.getElementById("time-horizon-input").value;
    const riskLevel = document.getElementById("risk-level-input").value;

    const prefs = Array.from(
      document.querySelectorAll("input[name='preferences']:checked")
    ).map((el) => el.value);

    const current = parseFloat(currentStr);
    const target = parseFloat(targetStr);
    const horizon = parseFloat(horizonStr);

    if (!goal) {
      setStatus("Please describe your goal.", "error");
      return;
    }
    if (!isFinite(current) || current <= 0) {
      setStatus("Please enter a valid current amount greater than 0.", "error");
      return;
    }
    if (!isFinite(target) || target <= 0) {
      setStatus("Please enter a valid target amount greater than 0.", "error");
      return;
    }
    if (!isFinite(horizon) || horizon <= 0) {
      setStatus("Please enter a valid time horizon in years.", "error");
      return;
    }
    if (!riskLevel) {
      setStatus("Please select your risk level.", "error");
      return;
    }

    const payload = {
      goal,
      current_amount: current,
      target_amount: target,
      time_horizon_years: horizon,
      risk_level: riskLevel,
      preferences: prefs
    };

    setStatus("Building strategy…", "loading");
    submitBtn.disabled = true;
    resultsEl.style.display = "none";

    try {
      const response = await fetch("/api/strategy-plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      let data;
      try {
        data = await response.json();
      } catch {
        data = null;
      }

      if (!response.ok || !data || data.error) {
        const msg = (data && data.error) ? data.error : `Server error: ${response.status}`;
        throw new Error(msg);
      }

      // --- Feasibility ---
      const feas = data.feasibility || {};
      const reqReturn = feas.required_annual_return;
      const label = feas.feasibility_label || "";
      const detail = feas.feasibility_explanation || "";

      let summaryText;
      if (reqReturn === null || reqReturn === undefined) {
        summaryText = "We could not compute a meaningful required annual return from the inputs.";
      } else {
        summaryText = "Required annual growth: " + formatPercent(reqReturn);
      }

      setFeasibilityBadge(label);
      feasibilitySummary.textContent = summaryText;
      feasibilityDetail.textContent = detail || "";

      // Risk warning
      riskWarningEl.textContent = data.risk_warning || "";

      // Growth chart
      if (data.growth_chart_image) {
        growthChartImg.src = "data:image/png;base64," + data.growth_chart_image;
        growthChartContainer.style.display = "block";
      } else {
        growthChartContainer.style.display = "none";
      }

      // --- Strategy & assets ---
      strategyTypeEl.textContent = data.strategy_type
        ? "Strategy type: " + data.strategy_type
        : "";

      strategyExplanationEl.textContent = data.explanation || "";

      // Recommended assets
      const assetsObj = data.recommended_assets || {};
      const assetCategories = Object.keys(assetsObj);
      if (assetCategories.length === 0) {
        assetsListEl.innerHTML = "<p>No specific asset examples were provided.</p>";
      } else {
        assetsListEl.innerHTML = assetCategories
          .map((cat) => {
            const items = assetsObj[cat] || [];
            const prettyCat = cat.charAt(0).toUpperCase() + cat.slice(1);
            const tags = items
              .map((name) => `<span class="asset-tag">${name}</span>`)
              .join(" ");
            return `
              <div style="margin-bottom: 6px;">
                <strong>${prettyCat}:</strong>
                <div class="asset-tags">${tags || "<span class='asset-tag'>No examples</span>"}</div>
              </div>
            `;
          })
          .join("");
      }

      // Examples "why"
      const examples = data.examples || [];
      if (!examples.length) {
        examplesListEl.innerHTML = "<li>No specific explanations were provided.</li>";
      } else {
        examplesListEl.innerHTML = examples
          .map((ex) => {
            const asset = ex.asset || "Asset";
            const why = ex.why || "";
            return `<li><strong>${asset}:</strong> ${why}</li>`;
          })
          .join("");
      }

      // Show results
      resultsEl.style.display = "grid";
      setStatus("Strategy generated ✔", "success");
    } catch (err) {
      console.error(err);
      setStatus("Error: " + err.message, "error");
      resultsEl.style.display = "none";
    } finally {
      submitBtn.disabled = false;
    }
  });
});
