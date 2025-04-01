from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import sys, pathlib, os, multiprocessing

# Set BASE_DIR and update sys.path to import backend modules outside the frontend folder
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR.parent))

# Import your backend function from dlc_runner
import dlc_runner

# Global variables
PROCESS = None
progress_value = 0   # Tracks progress from 0 to 100
final_csv = None     # Stores the final CSV path once processing completes

def dashboard_view(request):
    return render(request, 'tracking/dashboard.html')

def new_videos_view(request):
    return render(request, 'tracking/new_videos.html')

def livestream_view(request):
    return render(request, 'tracking/livestream.html')

@csrf_exempt
def process_video(request):
    """
    Starts a background process that runs the DLC pipeline.
    Resets global progress/csv, saves the uploaded video, and returns JSON.
    """
    global PROCESS, progress_value, final_csv
    if request.method == 'POST':
        uploaded_file = request.FILES.get('videos')
        if not uploaded_file:
            return JsonResponse({'error': 'No video file found'}, status=400)

        # Reset progress and final CSV path at the start
        progress_value = 0
        final_csv = None

        temp_video_path = os.path.join(BASE_DIR, 'temp_upload_' + uploaded_file.name)
        with open(temp_video_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        try:
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
    Terminates the running process (if any) and resets progress/csv.
    """
    global PROCESS, progress_value, final_csv
    if request.method == 'POST':
        if PROCESS is not None and PROCESS.is_alive():
            PROCESS.terminate()
            PROCESS.join()
            PROCESS = None
            progress_value = 0
            final_csv = None
            return JsonResponse({'status': 'Process stopped'})
        else:
            return JsonResponse({'status': 'No active process'})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)

def get_progress(request):
    """
    Returns current progress_value and final_csv (if progress is 100).
    The frontend polls this periodically to update the progress bar
    and trigger CSV download when completed.
    """
    global progress_value, final_csv
    return JsonResponse({
        'progress': progress_value,
        'csv_file': final_csv if progress_value >= 100 else None
    })
