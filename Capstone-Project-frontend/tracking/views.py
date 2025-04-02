# tracking/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import sys, pathlib, os, multiprocessing
from django.http import FileResponse, Http404
from django.urls import reverse
from tracking.progress_manager import get_shared_progress
from django.http import FileResponse
from django.conf import settings
from django.http import HttpResponse
import os

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR.parent))

# Import your backend function from dlc_runner (ensure dlc_runner does NOT import views.py)
import dlc_runner

# Import shared state from progress_manager instead of defining here
from tracking.progress_manager import get_shared_progress


PROCESS = None

def dashboard_view(request):
    return render(request, 'tracking/dashboard.html')

def new_videos_view(request):
    return render(request, 'tracking/new_videos.html')

def livestream_view(request):
    return render(request, 'tracking/livestream.html')

from django.http import FileResponse, Http404

def download_final_csv(request):
    global final_csv

    if not final_csv or not final_csv.value:
        raise Http404("No final CSV available.")

    try:
        return FileResponse(open(final_csv.value, 'rb'), as_attachment=True, filename='output_postprocessed.csv')
    except FileNotFoundError:
        raise Http404("CSV file not found on the server.")



from tracking.progress_manager import get_shared_progress

@csrf_exempt
def process_video(request):
    global PROCESS
    if request.method == 'POST':
        uploaded_file = request.FILES.get('videos')
        if not uploaded_file:
            return JsonResponse({'error': 'No video file found'}, status=400)

        #  Get shared manager values
        progress_value, final_csv = get_shared_progress()

        # Reset progress
        progress_value.value = 0
        final_csv.value = ""

        # Save uploaded video to a temp path
        temp_video_path = os.path.join(BASE_DIR, 'temp_upload_' + uploaded_file.name)
        with open(temp_video_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        try:
            #  Start the DLC processing in a subprocess
            PROCESS = multiprocessing.Process(
                target=dlc_runner.run_dlc_pipeline,
                args=(temp_video_path, progress_value, final_csv)
            )
            PROCESS.start()
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

        return JsonResponse({'status': 'Processing started'})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)


@csrf_exempt
def stop_video(request):
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
    progress_value, final_csv = get_shared_progress()

    if progress_value.value >= 100 and final_csv.value:
        return JsonResponse({
            'progress': progress_value.value,
            'csv_url': f'/download/{os.path.basename(final_csv.value)}'
        })

    return JsonResponse({
        'progress': progress_value.value,
        'csv_url': None
    })
@csrf_exempt
def download_file(request, filename):
    from tracking.progress_manager import get_shared_progress
    _, final_csv = get_shared_progress()

    if not final_csv.value or not os.path.exists(final_csv.value):
        raise Http404("File not found")

    return FileResponse(open(final_csv.value, 'rb'), as_attachment=True, filename=filename)

