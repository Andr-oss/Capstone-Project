import csv
import sys, pathlib, os, multiprocessing
import tempfile

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .run_visualization import run_visualization

# Set BASE_DIR and update sys.path to import backend modules outside the frontend folder
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR.parent))

# Import your backend function from dlc_runner
from tracking import dlc_runner

# Global variable to hold the background processing process
PROCESS = None

def dashboard_view(request):
    return render(request, 'tracking/dashboard.html')

def process_view(request):
    return render(request, 'tracking/Process.html')

def visualize_view(request):
    return render(request, 'tracking/visualize.html')

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


def run_visualize(request):
    if request.method == 'POST':
        # Get form parameters
        trail_length = int(request.POST.get('trail_length', 10))
        show_trails = request.POST.get('show_trails') == 'on'
        show_trajectory = request.POST.get('show_trajectory') == 'on'
        trajectory_opacity = float(request.POST.get('trajectory_opacity', 0.3))
        show_connections = request.POST.get('show_connections') == 'on'
        show_segmentation = request.POST.get('show_segmentation') == 'on'
        segmentation_opacity = float(request.POST.get('segmentation_opacity', 0.5))

        # Handle uploaded files
        video_files = request.FILES.getlist('videos')
        csv_files = request.FILES.getlist('csv_files')

        # Save uploaded files temporarily
        temp_video_paths = []
        temp_csv_paths = []

        for video in video_files:
            fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            with os.fdopen(fd, 'wb') as f:
                for chunk in video.chunks():
                    f.write(chunk)
            temp_video_paths.append(temp_path)

        for csv_file in csv_files:
            fd, temp_path = tempfile.mkstemp(suffix='.csv')
            with os.fdopen(fd, 'wb') as f:
                for chunk in csv_file.chunks():
                    f.write(chunk)
            temp_csv_paths.append(temp_path)

        try:
            # Call the visualization function with all parameters
            result = run_visualization(
                video_path=temp_video_paths,
                csv_path=temp_csv_paths,
                trail_length=trail_length,
                show_trails=show_trails,
                show_trajectory=show_trajectory,
                trajectory_opacity=trajectory_opacity,
                show_connections=show_connections,
                show_segmentation=show_segmentation,
                segmentation_opacity=segmentation_opacity
            )

            # Return the result
            return JsonResponse({'status': 'success', 'result': result})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
        finally:
            # Clean up temporary files
            for path in temp_video_paths + temp_csv_paths:
                try:
                    os.remove(path)
                except:
                    pass

    # If not POST, just render the form page
    return render(request, 'your_template.html')