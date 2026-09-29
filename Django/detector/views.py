import io
import ipaddress
import logging
import socket
from datetime import datetime
from urllib.parse import urljoin, urlparse
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)

from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .exceptions import (
    InvalidImageError,
    ModelNotAvailableError,
    PredictionError,
)
from .inference import (
    get_model_choices,
    initial_scan_ai_root,
    predict_two_stage,
    trained_ai_root,
)
from .models import ScanHistory

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
REMOTE_IMAGE_TIMEOUT_SECONDS = 12
MAX_REDIRECTS = 5


def get_session_key(request):
    if not request.session.session_key:
        request.session.create()
    return request.session.session_key


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
            "initial_scan_ai_root": initial_scan_ai_root(),
        },
    )


def result(request):
    return render(request, "detector/result.html", {"active_page": "result"})


def about(request):
    return render(request, "detector/about.html", {"active_page": "about"})


def dashboard(request):
    session_key = get_session_key(request)
    predictions = list(ScanHistory.objects.filter(session_key=session_key).order_by('-created_at'))
    healthy_scans = sum(1 for prediction in predictions if prediction.label == 'healthy')
    diseased_scans = sum(1 for prediction in predictions if prediction.label == 'black_sigatoka')
    
    # Calculate disease distribution percentages
    total = len(predictions)
    disease_distribution = []
    if total > 0:
        healthy_pct = round((healthy_scans / total) * 100)
        diseased_pct = round((diseased_scans / total) * 100)
        disease_distribution = [
            {'name': 'Healthy', 'pct': healthy_pct},
            {'name': 'Black Sigatoka', 'pct': diseased_pct},
        ]

    return render(
        request,
        'detector/dashboard.html',
        {
            'active_page': 'dashboard',
            'total_scans': len(predictions),
            'healthy_scans': healthy_scans,
            'diseased_scans': diseased_scans,
            'predictions': predictions,
            'disease_distribution': disease_distribution,
            'request': request,
        },
    )


@require_POST
def delete_scan(request, scan_id):
    session_key = get_session_key(request)
    scan = get_object_or_404(ScanHistory, pk=scan_id, session_key=session_key)
    scan.delete()
    return redirect('detector:dashboard')


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
        session_key = get_session_key(request)
        
        # Only save successful scans (stage 1 accepted)
        health_scan = prediction.get('health_scan')
        if health_scan:
            label = health_scan.get('label') or 'unknown'
            confidence = float(health_scan.get('confidence', 0.0) or 0.0)
            ScanHistory.objects.create(
                session_key=session_key,
                label=label,
                confidence=confidence,
                result=prediction,
            )
        
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


@require_POST
def get_scan_result(request, scan_id):
    """Retrieve full result data for a scan by ID."""
    session_key = get_session_key(request)
    scan = get_object_or_404(ScanHistory, pk=scan_id, session_key=session_key)
    return JsonResponse(scan.result)


def session_report(request):
    """Display a comprehensive report of all scans in the session."""
    session_key = get_session_key(request)
    all_scans = list(ScanHistory.objects.filter(session_key=session_key).order_by('-created_at'))
    
    total_scans = len(all_scans)
    healthy_scans = sum(1 for s in all_scans if s.label == 'healthy')
    diseased_scans = sum(1 for s in all_scans if s.label == 'black_sigatoka')
    
    # Calculate percentages
    healthy_pct = int((healthy_scans / total_scans * 100)) if total_scans > 0 else 0
    diseased_pct = int((diseased_scans / total_scans * 100)) if total_scans > 0 else 0
    
    # Build scan details for each scan
    scans_data = []
    for scan in all_scans:
        result_data = scan.result
        initial_scan = result_data.get('initial_scan', {})
        health_scan = result_data.get('health_scan', {})
        
        # Extract probabilities and format them
        probabilities = health_scan.get('probabilities', {})
        prob_list = []
        if probabilities:
            sorted_probs = sorted(probabilities.items(), key=lambda x: float(x[1]), reverse=True)
            for class_name, prob_value in sorted_probs:
                prob_float = float(prob_value)
                prob_pct = int(prob_float * 100 + 0.5)
                display_name = class_name.replace('_', ' ').title()
                prob_list.append((display_name, prob_pct))
        
        # Use stored confidence value
        confidence = round(scan.confidence * 100) if scan.confidence is not None else 0
        
        scan_info = {
            'id': scan.id,
            'date': scan.created_at.strftime("%b %d, %Y"),
            'time': scan.created_at.strftime("%I:%M %p"),
            'label': scan.label,
            'label_display': health_scan.get('label_display', 'Unknown').upper(),
            'confidence': confidence,
            'health_reliable': health_scan.get('reliable_match', False),
            'probabilities': prob_list,
        }
        scans_data.append(scan_info)
    
    return render(request, 'detector/session_report.html', {
        'total_scans': total_scans,
        'healthy_scans': healthy_scans,
        'diseased_scans': diseased_scans,
        'healthy_pct': healthy_pct,
        'diseased_pct': diseased_pct,
        'scans': scans_data,
    })

