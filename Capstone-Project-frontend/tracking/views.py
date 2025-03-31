from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, authenticate
import csv
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import sys, pathlib, os

# Set BASE_DIR and update sys.path to import backend modules outside the frontend folder
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR.parent))

# Import your backend function from dlc_runner (adjust the function name/path as needed)
import dlc_runner

def dashboard_view(request):
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'tracking/dashboard.html')

def new_videos_view(request):
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'tracking/new_videos.html')

def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()  # Saves the new user to the database (SQLite by default)
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            new_user = authenticate(username=username, password=raw_password)
            if new_user is not None:
                login(request, new_user)
                return redirect('dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'tracking/register.html', {'form': form})

def download_csv(request):
    # Create a HttpResponse object with CSV content
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="tracking_data.csv"'
    writer = csv.writer(response)
    writer.writerow([
        "Frame",
        "Middle_rat_X",
        "middle_rat_Y",
        "middle_rat_height",
        "middle_rat_width",
        "Head_rat_X",
        "Head_rat_Y",
        "Head_rat_height",
        "Head_rat_width"
    ])
    return response

def livestream_view(request):
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'tracking/livestream.html')

@csrf_exempt
def process_video(request):
    """
    Handles the uploaded video, calls the run_dlc_pipeline function (which includes post-processing),
    and returns a JSON response with the path of the final processed CSV.
    """
    if request.method == 'POST':
        uploaded_file = request.FILES.get('videos')
        if not uploaded_file:
            return JsonResponse({'error': 'No video file found'}, status=400)
        # Save the uploaded file temporarily
        temp_video_path = os.path.join(BASE_DIR, 'temp_upload_' + uploaded_file.name)
        with open(temp_video_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # You can pass a custom config_path if needed, or rely on the default defined in run_dlc_pipeline.
        try:
            output_csv = dlc_runner.run_dlc_pipeline(temp_video_path)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

        return JsonResponse({'status': 'Processing completed successfully', 'csv_file': output_csv})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)