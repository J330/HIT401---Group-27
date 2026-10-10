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
  const cameraDialog = document.getElementById("camera-dialog");
  const cameraVideo = document.getElementById("camera-video");
  const cameraCanvas = document.getElementById("camera-canvas");
  const cameraMessage = document.getElementById("camera-message");
  const cameraSnap = document.getElementById("camera-snap");
  const cameraNative = document.getElementById("camera-native");
  const cameraClose = document.getElementById("camera-close");
  const cameraCancel = document.getElementById("camera-cancel");

  const CHECK_NAMES = [
    ["type", "File type"], ["size", "File size"], ["resolution", "Resolution"],
    ["sharpness", "Sharpness"], ["exposure", "Lighting"], ["contrast", "Contrast"],
  ];
  const ICON = { pass: "✓", warn: "!", fail: "✕", pending: "", idle: "" };

  let selectedFile = null;
  let assessment = null;
  let busy = false;
  let cameraStream = null;

  if (USE_MOCK_API) mockNote.hidden = false;
  renderChecklist(null);

  /* ---------- File sources ---------- */

  chooseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  cameraBtn.addEventListener("click", (e) => { e.stopPropagation(); openCamera(); });
  changeBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  dropZone.addEventListener("click", () => { if (!selectedFile && !busy) fileInput.click(); });
  dropZone.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !selectedFile && e.target === dropZone) {
      e.preventDefault(); fileInput.click();
    }
  });

  /* ---------- Live camera ---------- */

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
      cameraStream = null;
    }
    cameraVideo.pause();
    cameraVideo.srcObject = null;
    cameraSnap.disabled = true;
  }

  function closeCamera() {
    if (cameraDialog.open) cameraDialog.close();
    stopCamera();
    cameraBtn.focus();
  }

  async function openCamera() {
    if (busy) return;
    if (!navigator.mediaDevices?.getUserMedia || !window.isSecureContext) {
      // Mobile browsers can launch their camera through a capture file input.
      // A desktop browser on a nonsecure host may only offer a file picker.
      cameraInput.click();
      return;
    }

    cameraDialog.showModal();
    cameraMessage.textContent = "Requesting camera permission…";
    cameraSnap.disabled = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: "environment" } },
      });
      // The user may have closed the dialog while permission was being requested.
      if (!cameraDialog.open) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      cameraStream = stream;
      cameraVideo.srcObject = stream;
      await cameraVideo.play();
      cameraSnap.disabled = false;
      cameraMessage.textContent = "Camera ready. Capture your photo when the leaf is in focus.";
    } catch (error) {
      stopCamera();
      const reason = error?.name === "NotAllowedError"
        ? "Camera permission was denied. Allow camera access in browser settings or use the device camera button."
        : error?.name === "NotFoundError"
          ? "No camera was found on this device. You can choose a photo instead."
          : "The live camera could not be opened. Try the device camera button or choose a photo.";
      cameraMessage.textContent = reason;
    }
  }

  cameraSnap.addEventListener("click", async () => {
    if (!cameraStream || !cameraVideo.videoWidth || !cameraVideo.videoHeight) return;
    cameraSnap.disabled = true;
    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    cameraCanvas.getContext("2d").drawImage(cameraVideo, 0, 0);
    try {
      const blob = await new Promise((resolve) => cameraCanvas.toBlob(resolve, "image/jpeg", 0.92));
      if (!blob) throw new Error("Could not capture the image.");
      const file = new File([blob], `leaf-photo-${Date.now()}.jpg`, { type: "image/jpeg" });
      closeCamera();
      await handleFile(file);
    } catch (_) {
      cameraMessage.textContent = "Could not save the captured photo. Please try again.";
      cameraSnap.disabled = false;
    }
  });

  cameraNative.addEventListener("click", () => {
    closeCamera();
    cameraInput.click();
  });
  cameraClose.addEventListener("click", closeCamera);
  cameraCancel.addEventListener("click", closeCamera);
  cameraDialog.addEventListener("close", stopCamera); // Escape also releases the camera.
  window.addEventListener("pagehide", stopCamera);
  cameraDialog.addEventListener("click", (e) => {
    if (e.target === cameraDialog) closeCamera();
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
        // Avoid failing the entire scan if two large image previews exceed
        // sessionStorage quota. The heatmap and other report data are retained.
        payload.image = null;
        if (payload.result) payload.result.background_removed_preview_url = null;
        try { sessionStorage.setItem("lastScan", JSON.stringify(payload)); }
        catch (_) { showStatus("The scan finished but the report is too large for browser storage. Try a smaller photo.", true); busy = false; return; }
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
