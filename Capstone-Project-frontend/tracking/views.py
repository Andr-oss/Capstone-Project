# tracking/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import sys, pathlib, os, multiprocessing
from django.http import FileResponse, Http404
from django.urls import reverse

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR.parent))

# Import your backend function from dlc_runner (ensure dlc_runner does NOT import views.py)
import dlc_runner

# Import shared state from progress_manager instead of defining here
from tracking.progress_manager import progress_value, final_csv

PROCESS = None

def dashboard_view(request):
    return render(request, 'tracking/dashboard.html')

def new_videos_view(request):
    return render(request, 'tracking/new_videos.html')

def livestream_view(request):
    return render(request, 'tracking/livestream.html')

def download_final_csv(request):
    """
    Serves the final CSV file as a downloadable response.
    """
    from tracking.progress_manager import final_csv  # or wherever you keep progress state

    # If there's no CSV path stored, raise a 404
    if not final_csv:
        raise Http404("No final CSV available.")

    try:
        return FileResponse(open(final_csv, 'rb'), as_attachment=True, filename='output_postprocessed.csv')
    except FileNotFoundError:
        raise Http404("CSV file not found on the server.")

@csrf_exempt
def process_video(request):
    global PROCESS, progress_value, final_csv
    if request.method == 'POST':
        uploaded_file = request.FILES.get('videos')
        if not uploaded_file:
            return JsonResponse({'error': 'No video file found'}, status=400)

        # Reset progress/csv at the start
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
    global progress_value, final_csv
    download_url = None
    # Only provide the download URL if the pipeline is finished and we have a final CSV path
    if progress_value >= 100 and final_csv:
        download_url = reverse('download_final_csv')  # e.g. "/download-final-csv/"

    return JsonResponse({
        'progress': progress_value,
        'csv_url': download_url,  # Return the URL for the frontend
    })
