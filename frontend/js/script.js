// frontend/js/script.js
// Small shared behaviour used on every page (underlines the current page's

document.addEventListener("DOMContentLoaded", () => {
  const current = window.location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll("nav a").forEach((link) => {
    if (link.getAttribute("href") === current) {
      link.style.textDecoration = "underline";
    }
  });
});
