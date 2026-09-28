/*
 * upload.html: file choice (picker, drag-and-drop, paste, camera),
 * quality checks, then submission and hand-off to result.html.
 */

document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const cameraInput = document.getElementById("camera-input");
  const chooseBtn = document.getElementById("choose-btn");
  const cameraBtn = document.getElementById("camera-btn");
  const changeBtn = document.getElementById("change-btn");
  const previewImg = document.getElementById("preview");
  const fileInfo = document.getElementById("file-info");
  const checklist = document.getElementById("checklist");
  const qualitySummary = document.getElementById("quality-summary");
  const submitBtn = document.getElementById("submit-btn");
  const statusLine = document.getElementById("status-line");
  const progress = document.getElementById("progress");
  const mockNote = document.getElementById("mock-note");

  const CHECK_NAMES = [
    ["type", "File type"], ["size", "File size"], ["resolution", "Resolution"],
    ["sharpness", "Sharpness"], ["exposure", "Lighting"], ["contrast", "Contrast"],
  ];
  const ICON = { pass: "✓", warn: "!", fail: "✕", pending: "", idle: "" };

  let selectedFile = null;
  let assessment = null;
  let busy = false;

  if (USE_MOCK_API) mockNote.hidden = false;
  renderChecklist(null);

  /* ---------- File sources ---------- */

  chooseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  cameraBtn.addEventListener("click", (e) => { e.stopPropagation(); cameraInput.click(); });
  changeBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  dropZone.addEventListener("click", () => { if (!selectedFile && !busy) fileInput.click(); });
  dropZone.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !selectedFile && e.target === dropZone) {
      e.preventDefault(); fileInput.click();
    }
  });

  fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));
  cameraInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

  ["dragenter", "dragover"].forEach((type) => dropZone.addEventListener(type, (e) => {
    e.preventDefault(); if (!busy) dropZone.classList.add("drag-over");
  }));
  ["dragleave", "drop"].forEach((type) => dropZone.addEventListener(type, (e) => {
    e.preventDefault(); dropZone.classList.remove("drag-over");
  }));
  dropZone.addEventListener("drop", (e) => { if (!busy) handleFile(e.dataTransfer.files[0]); });

  document.addEventListener("paste", (e) => {
    if (busy) return;
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) handleFile(item.getAsFile());
  });

  /* ---------- Quality assessment ---------- */

  async function handleFile(file) {
    if (!file) return;
    selectedFile = null;
    assessment = null;
    submitBtn.disabled = true;
    showStatus("");
    renderChecklist("pending");
    qualitySummary.textContent = "Checking photo quality…";
    qualitySummary.dataset.state = "pending";

    const result = await assessImage(file);
    fileInput.value = ""; cameraInput.value = "";

    if (result.previewDataUrl) {
      previewImg.src = result.previewDataUrl;
      fileInfo.textContent = `${file.name || "Pasted image"}, ${result.width} × ${result.height}`;
      dropZone.classList.add("has-file");
    } else {
      dropZone.classList.remove("has-file");
    }

    renderChecklist(result.checks);

    if (!result.ok) {
      const failed = result.checks.filter((c) => c.status === "fail").map((c) => c.name.toLowerCase());
      qualitySummary.textContent = `This photo can't be analysed: ${failed.join(", ")} check failed. Fix the issue marked in red and choose another photo.`;
      qualitySummary.dataset.state = "fail";
      return;
    }

    selectedFile = file;
    assessment = result;
    submitBtn.disabled = false;
    if (result.hasWarnings) {
      qualitySummary.textContent = "Ready to analyse. Some checks raised warnings, so treat the result with extra care or retake the photo.";
      qualitySummary.dataset.state = "warn";
    } else {
      qualitySummary.textContent = "All checks passed. Ready to analyse.";
      qualitySummary.dataset.state = "pass";
    }
  }

  function renderChecklist(state) {
    checklist.innerHTML = "";
    for (const [id, name] of CHECK_NAMES) {
      const c = Array.isArray(state) ? state.find((x) => x.id === id) : null;
      const status = c ? c.status : state === "pending" ? "pending" : "idle";
      const li = document.createElement("li");
      li.className = "check";
      li.dataset.status = status;
      li.innerHTML = `
        <span class="check-icon" aria-hidden="true">${ICON[status]}</span>
        <span class="check-name"></span>
        <span class="check-value"></span>
        <p class="check-msg"></p>`;
      li.querySelector(".check-name").textContent = name;
      li.querySelector(".check-value").textContent = c ? c.value : "";
      const statusWord = { pass: "Passed", warn: "Warning", fail: "Failed", pending: "Checking", idle: "Waiting for a photo" }[status];
      li.querySelector(".check-msg").textContent = c ? c.message : status === "idle" ? "Waiting for a photo." : "Checking…";
      li.setAttribute("aria-label", `${name}: ${statusWord}. ${c ? c.message : ""}`);
      checklist.appendChild(li);
    }
    // the decode check only exists when decoding fails
    const decode = Array.isArray(state) && state.find((x) => x.id === "decode");
    if (decode) qualitySummary.textContent = decode.message;
  }

  function showStatus(message, isError = false) {
    statusLine.textContent = message;
    statusLine.classList.toggle("error", isError);
  }

  /* ---------- Submission ---------- */

  const stepEls = [...progress.querySelectorAll("li")];
  let stepTimer = null;

  function runProgress() {
    progress.classList.add("is-active");
    let i = 0;
    stepEls.forEach((el) => el.classList.remove("is-running", "is-done"));
    stepEls[0].classList.add("is-running");
    // Stages advance on a timer as a guide; the final one waits for the real response.
    stepTimer = setInterval(() => {
      if (i >= stepEls.length - 2) return;
      stepEls[i].classList.replace("is-running", "is-done");
      i++;
      stepEls[i].classList.add("is-running");
    }, 900);
  }
  function stopProgress(success) {
    clearInterval(stepTimer);
    if (success) stepEls.forEach((el) => { el.classList.remove("is-running"); el.classList.add("is-done"); });
    else progress.classList.remove("is-active");
  }

  submitBtn.addEventListener("click", async () => {
    if (!selectedFile || !assessment?.ok || busy) return;
    busy = true;
    submitBtn.disabled = true;
    changeBtn.disabled = true;
    showStatus("");
    runProgress();

    try {
      const result = await predictImage(selectedFile);
      stopProgress(true);
      const payload = {
        result,
        image: assessment.previewDataUrl,
        fileName: selectedFile.name || "Pasted image",
        dimensions: [assessment.width, assessment.height],
        quality: assessment.checks,
        analysedAt: new Date().toISOString(),
      };
      try {
        sessionStorage.setItem("lastScan", JSON.stringify(payload));
      } catch (_) {
        payload.image = null; // storage full — show the result without the photo rather than fail
        sessionStorage.setItem("lastScan", JSON.stringify(payload));
      }
      window.location.href = "result.html";
    } catch (err) {
      stopProgress(false);
      showStatus(err.message || "Something went wrong. Try again.", true);
      submitBtn.disabled = false;
      changeBtn.disabled = false;
      busy = false;
    }
  });
});
