// script.js

document.addEventListener("DOMContentLoaded", () => {
    const runBtn = document.getElementById("run-btn");
    const statusEl = document.getElementById("status");
    const resultEl = document.getElementById("result");
    const companyInput = document.getElementById("company-input");

    runBtn.addEventListener("click", async () => {
        // Get company from input, default to AAPL if empty
        const company = (companyInput.value || "").trim() || "AAPL";

        // UI: reset + show loading state
        statusEl.textContent = `Running analysis for ${company}...`;
        resultEl.textContent = "";
        runBtn.disabled = true;

        try {
            // Call the Flask endpoint
            const response = await fetch(
                `/api/analyze-company?company=${encodeURIComponent(company)}`
            );

            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }

            const data = await response.json();

            // Expecting something like: { "summary": "..." }
            const summary =
                data.summary || data.result || JSON.stringify(data, null, 2);

            resultEl.textContent = summary || "No result returned.";
            statusEl.textContent = "Done!";
        } catch (err) {
            console.error("Error calling API:", err);
            statusEl.textContent = "Error running analysis.";
            resultEl.textContent = String(err);
        } finally {
            runBtn.disabled = false;
        }
    });
});



