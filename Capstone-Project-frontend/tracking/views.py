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
        try:
            # Get form parameters
            trail_length = int(request.POST.get('trail_length', 10))
            show_trails = request.POST.get('show_trails') == 'on'
            show_trajectory = request.POST.get('show_trajectory') == 'on'
            trajectory_opacity = float(request.POST.get('trajectory_opacity', 0.3))
            show_connections = request.POST.get('show_connections') == 'on'
            show_segmentation = request.POST.get('show_segmentation') == 'on'
            segmentation_opacity = float(request.POST.get('segmentation_opacity', 0.5))
            show_labels = request.POST.get('showLabels') == 'on'
            show_legend = request.POST.get('showLegend') == 'on'

            # Handle uploaded files
            video_file = request.FILES.get('videos')  # Get single file
            csv_file = request.FILES.get('csv_files')  # Get single file

            if not csv_file:
                return JsonResponse({'status': 'error', 'message': 'No CSV file provided'})

            # Save uploaded files temporarily
            temp_video_path = None
            temp_csv_path = None

            try:
                # Save CSV file
                fd, temp_csv_path = tempfile.mkstemp(suffix='.csv')
                with os.fdopen(fd, 'wb') as f:
                    for chunk in csv_file.chunks():
                        f.write(chunk)

                # Save video file if provided
                if video_file:
                    fd, temp_video_path = tempfile.mkstemp(suffix='.mp4')
                    with os.fdopen(fd, 'wb') as f:
                        for chunk in video_file.chunks():
                            f.write(chunk)

                # Call the visualization function with all parameters
                result = run_visualization(
                    csv_path=temp_csv_path,
                    video_path=temp_video_path,
                    trail_length=trail_length,
                    show_trails=show_trails,
                    show_trajectory=show_trajectory,
                    trajectory_opacity=trajectory_opacity,
                    show_connections=show_connections,
                    show_segmentation=show_segmentation,
                    segmentation_opacity=segmentation_opacity,
                    show_labels=show_labels,
                    show_legend=show_legend
                )

                return JsonResponse({'status': 'success', 'result': result})

            except Exception as e:
                return JsonResponse({'status': 'error', 'message': str(e)})
            finally:
                # Clean up temporary files
                if temp_csv_path and os.path.exists(temp_csv_path):
                    try:
                        os.remove(temp_csv_path)
                    except:
                        pass
                if temp_video_path and os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                    except:
                        pass

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Error processing request: {str(e)}'})

    # If not POST, render the visualization form
    return render(request, 'tracking/visualize.html')