/* Upload page: preview, online-image drag/drop and DINOv2 scan hand-off. */

document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const preview = document.getElementById("preview");
  const modelSelect = document.getElementById("model-select");
  const submitBtn = document.getElementById("submit-btn");
  const spinner = document.getElementById("spinner");
  const statusLine = document.getElementById("status-line");
  const loadingPanel = document.getElementById("loading-panel");
  const loadingBarFill = document.getElementById("loading-bar-fill");
  const loadingPercent = document.getElementById("loading-percent");
  const loadingLabel = document.getElementById("loading-label");

  const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
  const MAX_CANVAS_DIMENSION = 4096;

  const inferenceSteps = [
    { value: 40, message: "Stage 1: analysing the original image…" },
    { value: 56, message: "Stage 1: creating the background-filtered image…" },
    { value: 72, message: "Stage 1: combining both DINOv2 views…" },
    { value: 86, message: "Stage 2: analysing leaf health…" },
    { value: 94, message: "Rendering the result page…" },
  ];

  let selectedFile = null;
  let selectedImageUrl = null;
  let stageTimer = null;
  let currentProgress = 0;

  function updateSubmitState() {
    const stage1Ready = window.INITIAL_SCAN_AVAILABLE !== false;
    const hasImage = Boolean(selectedFile || selectedImageUrl);

    submitBtn.disabled =
      !stage1Ready ||
      !hasImage ||
      !modelSelect ||
      modelSelect.disabled ||
      !modelSelect.value;
  }

  function showStatus(message, isError = false) {
    statusLine.textContent = message;
    statusLine.classList.toggle("error", isError);
  }

  function setPreview(src) {
    preview.src = src;
    preview.style.display = "block";
  }

  function showLoading(show) {
    loadingPanel.style.display = show ? "block" : "none";
    spinner.style.display = show ? "block" : "none";
  }

  function setProgress(value, message = null) {
    currentProgress = Math.max(0, Math.min(100, Math.round(value)));
    loadingBarFill.style.width = `${currentProgress}%`;
    loadingPercent.textContent = `${currentProgress}%`;
    if (message) {
      loadingLabel.textContent = message;
    }
  }

  function stopStageSimulation() {
    if (stageTimer) {
      clearInterval(stageTimer);
      stageTimer = null;
    }
  }

  function startStageSimulation() {
    stopStageSimulation();
    let index = 0;

    stageTimer = window.setInterval(() => {
      if (index >= inferenceSteps.length) {
        stopStageSimulation();
        return;
      }

      const step = inferenceSteps[index];
      if (currentProgress < step.value) {
        setProgress(step.value, step.message);
      }
      index += 1;
    }, 1200);
  }

  async function imageFileToJpeg(file) {
    let bitmap = null;

    try {
      bitmap = await createImageBitmap(file);

      let width = bitmap.width;
      let height = bitmap.height;

      const longest = Math.max(width, height);
      if (longest > MAX_CANVAS_DIMENSION) {
        const scale = MAX_CANVAS_DIMENSION / longest;
        width = Math.max(1, Math.round(width * scale));
        height = Math.max(1, Math.round(height * scale));
      }

      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(bitmap, 0, 0, width, height);

      const blob = await new Promise((resolve, reject) => {
        canvas.toBlob(
          (value) => value ? resolve(value) : reject(new Error("Image conversion failed.")),
          "image/jpeg",
          0.94
        );
      });

      if (blob.size > MAX_UPLOAD_BYTES) {
        throw new Error("The converted image is larger than 10 MB.");
      }

      const baseName = (file.name || "online-image").replace(/\.[^.]+$/, "");
      return new File([blob], `${baseName}.jpg`, { type: "image/jpeg" });
    } finally {
      if (bitmap && typeof bitmap.close === "function") {
        bitmap.close();
      }
    }
  }

  async function setFile(file) {
    selectedFile = null;
    selectedImageUrl = null;

    if (window.INITIAL_SCAN_AVAILABLE === false) {
      showStatus(
        "Stage 1 is not ready. Build initial_references.pt first, then restart Django.",
        true
      );
      updateSubmitState();
      return;
    }

    if (!file || !String(file.type || "").startsWith("image/")) {
      showStatus("Please choose or drag an image file.", true);
      updateSubmitState();
      return;
    }

    if (file.size > MAX_UPLOAD_BYTES) {
      showStatus("Please choose an image smaller than 10 MB.", true);
      updateSubmitState();
      return;
    }

    try {
      selectedFile = await imageFileToJpeg(file);
      selectedImageUrl = null;

      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(selectedFile);

      showStatus("");
    } catch (error) {
      showStatus(
        error?.message || "The browser could not prepare this image for scanning.",
        true
      );
    }

    updateSubmitState();
  }

  function dataUrlToFile(dataUrl) {
    const parts = dataUrl.split(",");
    if (parts.length < 2) {
      throw new Error("Invalid image data.");
    }

    const header = parts[0];
    const mimeMatch = header.match(/^data:([^;,]+)/i);
    const mime = mimeMatch ? mimeMatch[1] : "image/png";

    let bytes;
    if (/;base64/i.test(header)) {
      const binary = atob(parts.slice(1).join(","));
      bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
      }
    } else {
      const text = decodeURIComponent(parts.slice(1).join(","));
      bytes = new TextEncoder().encode(text);
    }

    return new File([bytes], "dragged-online-image", { type: mime });
  }

  function extractImageUrl(dataTransfer) {
    const html = dataTransfer.getData("text/html");
    if (html) {
      try {
        const doc = new DOMParser().parseFromString(html, "text/html");
        const img = doc.querySelector("img");
        const src = img?.getAttribute("src") || img?.src;
        if (src) return src;
      } catch (_) {}
    }

    const uriList = dataTransfer.getData("text/uri-list");
    if (uriList) {
      const url = uriList
        .split(/\r?\n/)
        .map((line) => line.trim())
        .find((line) => line && !line.startsWith("#"));
      if (url) return url;
    }

    const plain = dataTransfer.getData("text/plain")?.trim();
    if (/^https?:\/\//i.test(plain) || /^data:image\//i.test(plain)) {
      return plain;
    }

    return null;
  }

  async function setOnlineImageUrl(url) {
    selectedFile = null;
    selectedImageUrl = null;

    if (!url) {
      showStatus("No usable image was found in the dropped item.", true);
      updateSubmitState();
      return;
    }

    if (/^data:image\//i.test(url)) {
      try {
        await setFile(dataUrlToFile(url));
      } catch (error) {
        showStatus(error?.message || "The dropped image could not be read.", true);
      }
      return;
    }

    if (!/^https?:\/\//i.test(url)) {
      showStatus("Please drag an image or a direct HTTP/HTTPS image.", true);
      updateSubmitState();
      return;
    }

    selectedImageUrl = url;
    setPreview(url);
    showStatus("Online image selected. Click Scan Image to analyse it.");
    updateSubmitState();
  }

  fileInput.addEventListener("change", async (e) => {
    await setFile(e.target.files[0]);
  });

  if (modelSelect) {
    modelSelect.addEventListener("change", updateSubmitState);
  }

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("drag-over");
    });
  });

  dropZone.addEventListener("drop", async (e) => {
    const files = e.dataTransfer?.files;

    if (files && files.length > 0 && String(files[0].type || "").startsWith("image/")) {
      await setFile(files[0]);
      return;
    }

    const url = extractImageUrl(e.dataTransfer);
    await setOnlineImageUrl(url);
  });

  dropZone.addEventListener("click", () => fileInput.click());

  submitBtn.addEventListener("click", async () => {
    if ((!selectedFile && !selectedImageUrl) || !modelSelect?.value) return;

    submitBtn.disabled = true;
    modelSelect.disabled = true;
    showLoading(true);
    setProgress(6, "Preparing scan…");
    showStatus("Running dual-view Banana Leaf validation (original + background-filtered), then health analysis...");
    startStageSimulation();

    try {
      const result = await predictImage(
        selectedFile,
        modelSelect.value,
        selectedImageUrl,
        {
          onUploadProgress(progress) {
            const value = 8 + Math.round(progress * 22);
            if (value > currentProgress && value < 35) {
              setProgress(value, "Uploading image to the server…");
            }
          },
        }
      );

      stopStageSimulation();
      setProgress(100, "Scan complete. Opening results…");
      sessionStorage.setItem("lastScanResult", JSON.stringify(result));
      sessionStorage.setItem("lastScanImage", preview.src);

      window.setTimeout(() => {
        window.location.href = window.RESULT_URL || "/result/";
      }, 220);
    } catch (err) {
      stopStageSimulation();
      showStatus(err.message || "Something went wrong. Please try again.", true);
      modelSelect.disabled = false;
      showLoading(false);
      setProgress(0, "Preparing scan…");
      updateSubmitState();
    }
  });

  updateSubmitState();
});
