// report.js

const fileInput   = document.getElementById("fileInput");
const analyzeBtn  = document.getElementById("analyzeBtn");
const statusEl    = document.getElementById("status");
const fileListEl  = document.getElementById("fileList");
const errorBox    = document.getElementById("errorBox");
const resultEl    = document.getElementById("result");
const resultTitle = document.getElementById("resultTitle");

// --- Helpers ---

function setStatus(msg) {
    if (statusEl) statusEl.textContent = msg || "";
}

function setError(msg) {
    if (!errorBox) return;
    if (msg) {
        errorBox.style.display = "block";
        errorBox.textContent = msg;
    } else {
        errorBox.style.display = "none";
        errorBox.textContent = "";
    }
}

function showFileList(files) {
    if (!fileListEl) return;
    if (!files || files.length === 0) {
        fileListEl.textContent = "";
        return;
    }
    const names = Array.from(files)
        .map(f => `• ${f.name}`)
        .join("\n");
    fileListEl.textContent = `Selected files:\n${names}`;
}

// --- Main handler ---

async function handleAnalyzeClick() {
    setError("");
    setStatus("");
    resultEl.textContent = "";
    if (resultTitle) resultTitle.style.display = "none";

    const files = fileInput.files;
    if (!files || files.length === 0) {
        setError("Please select at least one report file first.");
        return;
    }

    showFileList(files);

    const formData = new FormData();
    Array.from(files).forEach(f => {
        // name must match server: request.files.getlist("files")
        formData.append("files", f);
    });

    analyzeBtn.disabled = true;
    setStatus("Analyzing reports with FinRobot + ChatGPT...");

    try {
        const resp = await fetch("/api/analyze-report", {
            method: "POST",
            body: formData
        });

        const data = await resp.json();

        if (!resp.ok) {
            const msg = data.error || `Request failed with status ${resp.status}`;
            setError(msg);
            setStatus("");
            return;
        }

        const analysis = data.analysis || "(No analysis text returned.)";

        if (resultTitle) {
            resultTitle.style.display = "block";
        }
        resultEl.textContent = analysis;
        setStatus("Analysis complete.");
    } catch (err) {
        console.error("Error calling /api/analyze-report:", err);
        setError("Unexpected error while contacting the server. Check the console for details.");
        setStatus("");
    } finally {
        analyzeBtn.disabled = false;
    }
}

// --- Wire up events ---

if (analyzeBtn) {
    analyzeBtn.addEventListener("click", handleAnalyzeClick);
}

if (fileInput) {
    fileInput.addEventListener("change", () => {
        showFileList(fileInput.files);
        setError("");
        setStatus("");
    });
}

