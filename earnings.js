// earnings.js

document.addEventListener("DOMContentLoaded", () => {
  const filesInput = document.getElementById("filesInput");
  const questionsInput = document.getElementById("questionsInput");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const outputDiv = document.getElementById("output");
  const statusSpan = document.getElementById("status");

  // Safety check so no null.innerHTML errors occur
  if (!filesInput || !questionsInput || !analyzeBtn || !outputDiv || !statusSpan) {
    console.error(
      "Earnings Q&A: Missing one or more required HTML elements. " +
      "Expected IDs: filesInput, questionsInput, analyzeBtn, output, status."
    );
    return;
  }

  // Helper to display status text
  function setStatus(msg) {
    statusSpan.textContent = msg || "";
  }

  // Helper to update output box
  function setOutput(html, { isEmpty = false } = {}) {
    outputDiv.innerHTML = html;
    if (isEmpty) {
      outputDiv.classList.add("empty");
    } else {
      outputDiv.classList.remove("empty");
    }
  }

  // Event listener for analyze button
  analyzeBtn.addEventListener("click", async (event) => {
    event.preventDefault();

    const files = filesInput.files;
    const questions = questionsInput.value.trim();

    // Reset statuses
    setStatus("");

    // Validate inputs
    if (!files || files.length === 0) {
      setStatus("Please upload at least one transcript file (PDF or TXT).");
      return;
    }

    if (!questions) {
      setStatus("Please enter one or more questions about the earnings call.");
      return;
    }

    // Build form data
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }
    formData.append("questions", questions);

    try {
      setStatus("Analyzing call and answering your questions…");
      setOutput("Processing…", { isEmpty: false });

      const response = await fetch("/api/earnings-qa", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        console.error("Error from server:", text);

        setStatus("Server error while analyzing earnings call.");
        setOutput("An error occurred while processing your request.", {
          isEmpty: false,
        });
        return;
      }

      const data = await response.json();

      if (data.error) {
        setStatus(data.error);
        setOutput("An error occurred while processing your request.", {
          isEmpty: false,
        });
        return;
      }

      const answers = data.answers || "No response returned from the model.";

      // Convert line breaks to paragraphs to look clean on UI
      const html = answers
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line.length > 0)
        .map((line) => `<p>${line}</p>`)
        .join("");

      setOutput(html || "No content returned.", { isEmpty: !html });
      setStatus("Done.");
    } catch (err) {
      console.error("Network Error:", err);
      setStatus("Network error while contacting the server.");
      setOutput("Could not reach the server. Make sure Flask is running.", {
        isEmpty: false,
      });
    }
  });
});


