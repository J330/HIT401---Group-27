// frontend/js/api.js
// Shared config + a single function for calling the backend API.

const API_BASE_URL = "http://127.0.0.1:8000";

async function predictImage(file) {
  const formData = new FormData();
  formData.append("image", file);

  const response = await fetch(`${API_BASE_URL}/api/predict/`, {
    method: "POST",
    headers: getAuthHeaders(),
    credentials: "include",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
  }
  return response.json();
}

function getAuthHeaders() {
  const token = localStorage.getItem("authToken");
  return token ? { Authorization: `Token ${token}` } : {};
}
