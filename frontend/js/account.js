const ACCOUNT_API_URL = "http://127.0.0.1:8000/api/login/";
const usernameInput = document.getElementById("username");
const passwordInput = document.getElementById("password");
const accountStatus = document.getElementById("accountStatus");

function getCookie(name) {
  return document.cookie
    .split("; ")
    .find((cookie) => cookie.startsWith(`${name}=`))
    ?.split("=")[1];
}

// Logs into the predefined local account and stores its API token.
async function loginAccount() {
  accountStatus.textContent = "Logging in...";
  document.getElementById("loginBtn").disabled = true;

  await fetch("http://127.0.0.1:8000/api/csrf/", { credentials: "include" });
  const csrfToken = getCookie("csrftoken");

  const response = await fetch(ACCOUNT_API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": csrfToken,
    },
    credentials: "include",
    body: JSON.stringify({
      username: usernameInput.value,
      password: passwordInput.value,
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Login failed.");

  localStorage.setItem("authToken", data.token);
  accountStatus.textContent = "Logged in. Opening home page...";
  window.location.href = "http://127.0.0.1:8080/index.html";
}

document.getElementById("loginBtn").addEventListener("click", () => {
  loginAccount().catch((error) => {
    accountStatus.textContent = error.message;
    document.getElementById("loginBtn").disabled = false;
  });
});