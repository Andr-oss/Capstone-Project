from django.shortcuts import render
import csv
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import sys, pathlib, os

# Set BASE_DIR and update sys.path to import backend modules outside the frontend folder
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR.parent))

# Import your backend function from dlc_runner
import dlc_runner

def dashboard_view(request):
    return render(request, 'tracking/dashboard.html')

def new_videos_view(request):
    return render(request, 'tracking/new_videos.html')

def download_csv(request):
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
        try:
            output_csv = dlc_runner.run_dlc_pipeline(temp_video_path)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        return JsonResponse({'status': 'Processing completed successfully', 'csv_file': output_csv})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)
