const predictionsUrl = "http://127.0.0.1:8000/api/predictions/";
const totalScans = document.querySelector(".summary-card:nth-child(1) .summary-number");
const healthyScans = document.querySelector(".summary-card.healthy .summary-number");
const diseasedScans = document.querySelector(".summary-card.diseased .summary-number");
const scanRows = document.querySelector(".scan-table tbody");

// Protects displayed values before adding them to the table.
function escapeHtml(value) {
	return String(value)
		.replaceAll("&", "&amp;")
		.replaceAll("<", "&lt;")
		.replaceAll(">", "&gt;")
		.replaceAll('"', "&quot;")
		.replaceAll("'", "&#039;");
}

// Converts the database label into readable text.
function formatLabel(label) {
	if (label === "healthy") return "Healthy";
	if (label === "black_sigatoka") return "Black Sigatoka";
	if (label === "not_black_sigatoka") return "Not Black Sigatoka";
	return label.replaceAll("_", " ");
}

// Shows a short explanation for the result label.
function formatDetail(label) {
	return label === "healthy"
		? "No visible lesions, spotting, or discoloration."
		: "Black Sigatoka detected.";
}

// Displays each saved result in the history table.
function renderRows(predictions) {
	if (predictions.length === 0) {
		scanRows.innerHTML = '<tr><td colspan="5">No scans yet.</td></tr>';
		return;
	}

	scanRows.innerHTML = predictions.map((prediction) => {
		const isHealthy = prediction.label === "healthy";
		const statusClass = isHealthy ? "healthy" : "diseased";
		const image = prediction.image
			? `<img class="scan-image" src="${escapeHtml(prediction.image)}" alt="Uploaded leaf">`
			: "";
		const date = new Date(prediction.created_at).toLocaleDateString(undefined, {
			year: "numeric", month: "short", day: "numeric"
		});

		return `
			<tr>
				<td class="col-image">${image}</td>
				<td class="col-status"><span class="status-pill ${statusClass}">${escapeHtml(formatLabel(prediction.label))}</span></td>
				<td class="col-detail">${escapeHtml(formatDetail(prediction.label))}</td>
				<td class="col-confidence">${(prediction.confidence * 100).toFixed(1)}%</td>
				<td class="col-date">${escapeHtml(date)}</td>
				<td class="col-actions">
					<button class="action-button delete delete-button" type="button" data-id="${prediction.id}">Delete</button>
				</td>
			</tr>`;
	}).join("");
}

// Loads results and updates the dashboard summary numbers.
async function loadDashboard() {
	const response = await fetch(predictionsUrl, { headers: getAuthHeaders(), credentials: "include" });
	if (!response.ok) throw new Error(`Unable to load history (${response.status})`);

	const predictions = await response.json();
	totalScans.textContent = predictions.length;
	healthyScans.textContent = predictions.filter(({ label }) => label === "healthy").length;
	diseasedScans.textContent = predictions.filter(({ label }) => label === "black_sigatoka").length;
	renderRows(predictions);
}

// Deletes the selected result from the history.
async function deletePrediction(id) {
	if (!window.confirm("Delete this scan from the history?")) return;

	const response = await fetch(`${predictionsUrl}${id}/`, { method: "DELETE", headers: getAuthHeaders(), credentials: "include" });
	if (!response.ok) throw new Error(`Unable to delete result (${response.status})`);
	await loadDashboard();
}

// Handles clicks on the edit and delete buttons.
scanRows.addEventListener("click", async (event) => {
	const button = event.target.closest("button[data-id]");
	if (!button) return;

	try {
		if (button.classList.contains("delete-button")) {
			await deletePrediction(button.dataset.id);
		}
	} catch (error) {
		window.alert(error.message);
	}
});

loadDashboard().catch((error) => {
	scanRows.innerHTML = `<tr><td colspan="5">${escapeHtml(error.message)}</td></tr>`;
});
