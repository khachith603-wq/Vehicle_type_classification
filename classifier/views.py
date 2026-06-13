from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Count, Avg
from django.utils import timezone
from datetime import timedelta
from collections import Counter

from .models import Prediction
from .utils import VehicleClassifier


def home(request):
    return render(request, 'classifier/home.html')


def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f"Welcome {user.username}! Your account has been created.")
            return redirect('dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'classifier/signup.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = authenticate(
                username=form.cleaned_data.get('username'),
                password=form.cleaned_data.get('password'),
            )
            if user is not None:
                login(request, user)
                return redirect('dashboard')
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    return render(request, 'classifier/login.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('home')


@login_required
def dashboard(request):
    predictions = Prediction.objects.filter(user=request.user)

    total_count    = predictions.count()
    avg_confidence = predictions.exclude(label="Error").aggregate(Avg('confidence'))['confidence__avg'] or 0

    # Top vehicle type from the new vehicle_type field
    type_counts = (
        predictions.exclude(vehicle_type__in=["Unknown", "Unknown / Non-Vehicle", "Error"])
        .values('vehicle_type')
        .annotate(cnt=Count('vehicle_type'))
        .order_by('-cnt')
    )
    top_type = type_counts[0]['vehicle_type'] if type_counts else "N/A"

    last_24h = predictions.filter(
        created_at__gte=timezone.now() - timedelta(days=1)
    ).count()

    # Vehicle type breakdown for the mini chart
    type_breakdown = list(type_counts[:6])

    context = {
        'predictions':      predictions[:4],
        'total_count':      total_count,
        'avg_confidence':   round(avg_confidence, 1),
        'top_type':         top_type,
        'last_24h':         last_24h,
        'type_breakdown':   type_breakdown,
    }
    return render(request, 'classifier/dashboard.html', context)


@login_required
def history_view(request):
    predictions = Prediction.objects.filter(user=request.user)
    # Build unique vehicle type list for the filter dropdown
    vehicle_types = (
        predictions.exclude(vehicle_type="")
        .values_list('vehicle_type', flat=True)
        .distinct()
        .order_by('vehicle_type')
    )

    # Optional filter by vehicle type
    filter_type = request.GET.get('type', '')
    if filter_type:
        predictions = predictions.filter(vehicle_type=filter_type)

    return render(request, 'classifier/history.html', {
        'predictions':   predictions,
        'vehicle_types': vehicle_types,
        'filter_type':   filter_type,
    })


@login_required
def classify_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        prediction_obj = Prediction.objects.create(
            user=request.user,
            image=image_file,
            label="Processing...",
            confidence=0.0,
            vehicle_type="Processing...",
        )

        try:
            prediction_obj.image.open('rb')
            label, confidence, vehicle_type, stage = VehicleClassifier.classify_image(
                prediction_obj.image
            )
            prediction_obj.image.close()

            prediction_obj.label        = label
            prediction_obj.confidence   = confidence
            prediction_obj.vehicle_type = vehicle_type
            prediction_obj.save()

            stage_note = " (specialist model)" if stage == "specialist" else " (general model)"
            messages.success(
                request,
                f"Classified as {vehicle_type}: {label} ({confidence:.1f}% confidence){stage_note}"
            )
        except Exception as e:
            prediction_obj.label        = "Error"
            prediction_obj.vehicle_type = "Error"
            prediction_obj.save()
            messages.error(request, f"Error processing image: {str(e)}")
        finally:
            try:
                prediction_obj.image.close()
            except Exception:
                pass

        return redirect('dashboard')

    return render(request, 'classifier/classify.html')


def about_view(request):
    return render(request, 'classifier/about.html')


def help_view(request):
    return render(request, 'classifier/help.html')
