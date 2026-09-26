import io
import ipaddress
import logging
import socket
from urllib.parse import urljoin, urlparse
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .inference import (
    InvalidImageError,
    ModelNotAvailableError,
    PredictionError,
    get_model_choices,
    predict_two_stage,
    trained_ai_root,
)

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
REMOTE_IMAGE_TIMEOUT_SECONDS = 12
MAX_REDIRECTS = 5


def home(request):
    return render(request, "detector/index.html", {"active_page": "home"})


def upload(request):
    model_choices = get_model_choices()
    initial_scan_available = (
        bool(model_choices) and bool(model_choices[0].get("initial_available"))
    )
    return render(
        request,
        "detector/upload.html",
        {
            "active_page": "upload",
            "model_choices": model_choices,
            "any_model_available": any(
                choice["health_available"] for choice in model_choices
            ),
            "initial_scan_available": initial_scan_available,
            "trained_ai_root": trained_ai_root(),
        },
    )


def result(request):
    return render(request, "detector/result.html", {"active_page": "result"})


def _validate_public_http_url(url: str) -> str:
    parsed = urlparse(url)

    if parsed.scheme.lower() not in {"http", "https"}:
        raise PredictionError("Only HTTP or HTTPS online images are supported.")

    if not parsed.hostname:
        raise PredictionError("The dropped online image URL is invalid.")

    if parsed.username or parsed.password:
        raise PredictionError("URLs containing usernames or passwords are not supported.")

    try:
        addresses = {
            info[4][0]
            for info in socket.getaddrinfo(
                parsed.hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
    except socket.gaierror as error:
        raise PredictionError("The online image host could not be found.") from error

    if not addresses:
        raise PredictionError("The online image host could not be found.")

    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise PredictionError("The online image host returned an invalid address.")

        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise PredictionError("Private or local network image URLs are not allowed.")

    return url


class _SafeRedirectHandler(HTTPRedirectHandler):
    def __init__(self):
        super().__init__()
        self.redirect_count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.redirect_count += 1
        if self.redirect_count > MAX_REDIRECTS:
            raise PredictionError("The online image redirected too many times.")

        absolute = urljoin(req.full_url, newurl)
        _validate_public_http_url(absolute)
        return super().redirect_request(req, fp, code, msg, headers, absolute)


def _download_remote_image(url: str) -> io.BytesIO:
    url = _validate_public_http_url(url)

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/153 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/png,image/jpeg,image/*;q=0.8,*/*;q=0.2",
        },
    )

    opener = build_opener(_SafeRedirectHandler())

    try:
        with opener.open(request, timeout=REMOTE_IMAGE_TIMEOUT_SECONDS) as response:
            data = response.read(MAX_UPLOAD_BYTES + 1)
    except PredictionError:
        raise
    except Exception as error:
        raise PredictionError(
            "The online image could not be downloaded. "
            "Try dragging the image itself, opening the full-size image first, "
            "or save it to your computer and upload it."
        ) from error

    if len(data) > MAX_UPLOAD_BYTES:
        raise PredictionError("The online image is larger than 10 MB.")

    if not data:
        raise PredictionError("The online image download was empty.")

    stream = io.BytesIO(data)
    stream.name = "online-image"
    return stream


@require_POST
def predict_api(request):
    uploaded_image = request.FILES.get("image")
    image_url = (request.POST.get("image_url") or "").strip()
    model_key = (request.POST.get("model") or "").strip()

    if uploaded_image is None and not image_url:
        return JsonResponse({"error": "Please upload or drag an image."}, status=400)

    if not model_key:
        return JsonResponse({"error": "Please choose a health analysis model."}, status=400)

    if uploaded_image is not None and uploaded_image.size > MAX_UPLOAD_BYTES:
        return JsonResponse(
            {"error": "The image is too large. Please use an image smaller than 10 MB."},
            status=400,
        )

    try:
        image_source = (
            uploaded_image
            if uploaded_image is not None
            else _download_remote_image(image_url)
        )

        prediction = predict_two_stage(image_source, model_key)
        return JsonResponse(prediction)

    except InvalidImageError as error:
        return JsonResponse({"error": str(error)}, status=400)
    except ModelNotAvailableError as error:
        return JsonResponse({"error": str(error)}, status=503)
    except PredictionError as error:
        return JsonResponse({"error": str(error)}, status=400)
    except Exception:
        logger.exception("Unexpected two-stage model inference failure")
        return JsonResponse(
            {
                "error": (
                    "The AI could not analyse this image. "
                    "Check the server terminal for the detailed error."
                )
            },
            status=500,
        )
