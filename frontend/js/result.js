// frontend/js/result.js
// Reads the prediction saved by upload.js.
const raw = sessionStorage.getItem("predictionResult");
const container = document.getElementById("resultContainer");

if (!raw) {
  container.innerHTML = "<p>No result found — please upload an image first.</p>";
} else {
  const result = JSON.parse(raw);
  const labelText = result.label === "not_black_sigatoka"
    ? "Not Black Sigatoka — could be something else"
    : result.label.replace("_", " ");

  container.innerHTML = `
    <p><strong>Result:</strong> ${labelText}</p>
    <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
  `;
}
