// forecast.js

let chart = null;

async function runScenarioForecast(event) {
  event.preventDefault();

  const companyInput = document.getElementById("companyInput");
  const scenarioInput = document.getElementById("scenarioInput");
  const statusText = document.getElementById("statusText");
  const runBtn = document.getElementById("runForecastBtn");
  const explanationEl = document.getElementById("explanation");
  const chartTitle = document.getElementById("chartTitle");

  const company = companyInput.value.trim();
  const scenario = scenarioInput.value.trim();

  if (!company || !scenario) {
    statusText.textContent = "Please enter both a company and a scenario.";
    statusText.classList.add("error");
    return;
  }

  statusText.classList.remove("error");
  statusText.textContent = "Running forecast… this may take a few seconds.";
  runBtn.disabled = true;
  explanationEl.textContent = "";
  chartTitle.textContent = "Loading…";

  try {
    const response = await fetch("/api/forecast_scenario", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ company, scenario }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Server error: ${response.status}`);
    }

    const data = await response.json();

    const historical = data.historical || [];
    const forecast = data.forecast || [];
    const explanation = data.explanation || "";
    const symbol = data.symbol || company.toUpperCase();
    const companyName = data.company || company;

    if (historical.length === 0) {
      statusText.textContent = "No historical data returned for that company.";
      statusText.classList.add("error");
      chartTitle.textContent = "No data loaded";
      return;
    }

    // Build labels as dates across historical + forecast
    const labels = [
      ...historical.map((p) => p.date),
      ...forecast.map((p) => p.date),
    ];

    const historicalMap = new Map(
      historical.map((p) => [p.date, p.close])
    );

    const forecastMap = new Map(
      forecast.map((p) => [p.date, p.price])
    );

    const historicalData = labels.map((d) =>
      historicalMap.has(d) ? historicalMap.get(d) : null
    );

    const forecastData = labels.map((d) =>
      forecastMap.has(d) ? forecastMap.get(d) : null
    );

    const ctx = document.getElementById("forecastChart").getContext("2d");

    if (chart) {
      chart.destroy();
    }

    chart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Historical (last 12 months)",
            data: historicalData,
            borderColor: "rgb(46, 204, 113)", // green
            backgroundColor: "rgba(46, 204, 113, 0.15)",
            tension: 0.25,
            spanGaps: true,
          },
          {
            label: "Scenario forecast (next 3 months)",
            data: forecastData,
            borderColor: "rgb(231, 76, 60)", // red
            backgroundColor: "rgba(231, 76, 60, 0.12)",
            tension: 0.25,
            spanGaps: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "index",
          intersect: false,
        },
        plugins: {
          legend: {
            labels: {
              color: "#f5f7ff",
            },
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                const value = context.parsed.y;
                if (value == null) return "";
                return `${context.dataset.label}: $${value.toFixed(2)}`;
              },
            },
          },
        },
        scales: {
          x: {
            ticks: {
              color: "#b7bfd9",
              maxRotation: 45,
              minRotation: 0,
            },
            grid: {
              color: "rgba(255, 255, 255, 0.04)",
            },
          },
          y: {
            ticks: {
              color: "#b7bfd9",
              callback: (val) => `$${val}`,
            },
            grid: {
              color: "rgba(255, 255, 255, 0.04)",
            },
          },
        },
      },
    });

    chartTitle.textContent = `${companyName} (${symbol}) – Scenario Forecast`;
    explanationEl.textContent = explanation;
    statusText.textContent = "Forecast generated successfully.";
  } catch (err) {
    console.error(err);
    statusText.textContent = `Error: ${err.message}`;
    statusText.classList.add("error");
    chartTitle.textContent = "Error loading data";
  } finally {
    runBtn.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("forecastForm");
  form.addEventListener("submit", runScenarioForecast);
});

