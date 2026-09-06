// frontend/js/api.js
// Shared config + a single function for calling the backend API.
// Change API_BASE_URL to your deployed Render backend URL.
const API_BASE_URL = "http://127.0.0.1:8000";

async function predictImage(file) {
  const formData = new FormData();
  formData.append("image", file);

  const response = await fetch(`${API_BASE_URL}/api/predict/`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
  }
  return response.json();
}
