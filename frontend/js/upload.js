// frontend/js/upload.js
// Previews the chosen image, then submits it and hands the result on result.html

document.getElementById("imageInput").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const img = document.getElementById("preview");
    img.src = reader.result;
    img.style.display = "block";
  };
  reader.readAsDataURL(file);
});

document.getElementById("uploadBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("imageInput");
  const statusMsg = document.getElementById("statusMsg");
  const file = fileInput.files[0];

  if (!file) {
    statusMsg.textContent = "Please choose an image first.";
    return;
  }

  statusMsg.textContent = "Analysing...";
  try {
    const result = await predictImage(file);
    sessionStorage.setItem("predictionResult", JSON.stringify(result));
    window.location.href = "result.html";
  } catch (err) {
    statusMsg.textContent = "Something went wrong: " + err.message;
  }
});
