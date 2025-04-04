from django.shortcuts import render
import csv
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import sys, pathlib, os, multiprocessing

# Set BASE_DIR and update sys.path to import backend modules outside the frontend folder
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR.parent))

# Import your backend function from dlc_runner
from tracking import dlc_runner

# Global variable to hold the background processing process
PROCESS = None

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
    Handles the uploaded video, starts the run_dlc_pipeline function in a background process,
    and returns a JSON response indicating that processing has started.
    """
    global PROCESS
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
            # Start the processing in a separate process
            PROCESS = multiprocessing.Process(
                target=dlc_runner.run_dlc_pipeline,
                args=(temp_video_path,)
            )
            PROCESS.start()
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        return JsonResponse({'status': 'Processing started'})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def stop_video(request):
    """
    Handles the stop process request.
    Terminates the background process running run_dlc_pipeline.
    """
    global PROCESS
    if request.method == 'POST':
        if PROCESS is not None and PROCESS.is_alive():
            PROCESS.terminate()
            PROCESS.join()
            PROCESS = None
            return JsonResponse({'status': 'Process stopped'})
        else:
            return JsonResponse({'status': 'No active process'})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)
