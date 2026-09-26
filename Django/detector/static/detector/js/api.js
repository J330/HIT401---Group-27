/* Django API client for the two-stage banana leaf scan. */

function predictImage(file, modelName, imageUrl = null, options = {}) {
  const formData = new FormData();

  if (file) {
    formData.append("image", file);
  } else if (imageUrl) {
    formData.append("image_url", imageUrl);
  }

  formData.append("model", modelName);

  const csrfToken = document.querySelector(
    "input[name='csrfmiddlewaretoken']"
  )?.value;

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", window.PREDICT_URL || "/api/predict/");

    if (csrfToken) {
      xhr.setRequestHeader("X-CSRFToken", csrfToken);
    }

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && typeof options.onUploadProgress === "function") {
        options.onUploadProgress(event.loaded / event.total);
      }
    };

    xhr.onerror = () => {
      reject(new Error("Couldn't reach the Django server. Check that it is running and try again."));
    };

    xhr.onload = () => {
      let body = null;
      try {
        body = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch (_) {
        body = null;
      }

      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body);
        return;
      }

      reject(new Error(body?.error || "Something went wrong while the AI analysed this image."));
    };

    xhr.send(formData);
  });
}
