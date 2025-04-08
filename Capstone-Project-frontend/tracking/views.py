import csv
import sys, pathlib, os, multiprocessing, time, subprocess
import tempfile
import json
import shutil
import os

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, FileResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from tracking.run_vis import run_visualization_to_client



# Set BASE_DIR and update sys.path to import backend modules outside the frontend folder
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR.parent))

# Import your backend function from dlc_runner
from tracking import dlc_runner

# Global variables
PROCESS = None
VIS_PROCESS = None  # For visualization process tracking
PROCESSING_STATUS = {"status": "idle", "csv_file": None}  # Track processing status


def dashboard_view(request):
    return render(request, 'tracking/dashboard.html')


def process_view(request):
    return render(request, 'tracking/Process.html')


def visualize_view(request):
    return render(request, 'tracking/visualize.html')


def download_csv(request):
    # Check if specific file was requested
    file_path = request.GET.get('file')

    if file_path and os.path.exists(file_path):
        # Return the specific file
        return FileResponse(open(file_path, 'rb'),
                            as_attachment=True,
                            filename=os.path.basename(file_path))
    else:
        # Fallback to the general CSV template
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


def check_visualization_status(request):
    """Check the status of the visualization process"""
    status_file = os.path.join(BASE_DIR, 'temp_visualizations', 'vis_status.json')
    try:
        with open(status_file, 'r') as f:
            status = json.load(f)
            return JsonResponse(status)
    except:
        return JsonResponse({'status': 'unknown'})


@csrf_exempt
def check_processing_status(request):
    """Returns the current status of the video processing"""
    global PROCESSING_STATUS
    return JsonResponse(PROCESSING_STATUS)


# Helper function to run DLC in a separate process and track status
def run_dlc_and_track_status(video_path, output_directory=None):
    global PROCESSING_STATUS
    try:
        PROCESSING_STATUS = {"status": "processing", "csv_file": None}
        # Run the DLC pipeline with the specified output directory
        output_csv = dlc_runner.run_dlc_pipeline(video_path, output_directory=output_directory)
        # Update status when complete
        PROCESSING_STATUS = {"status": "complete", "csv_file": output_csv}
    except Exception as e:
        PROCESSING_STATUS = {"status": "error", "message": str(e)}
        # Make sure to clean up the temp file if there's an error
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except:
            pass


@csrf_exempt
def process_video(request):
    """
    Handles the uploaded video synchronously, processes it, and immediately
    returns the resulting CSV file as a download response.
    """
    if request.method == 'POST':
        uploaded_file = request.FILES.get('videos')
        if not uploaded_file:
            return HttpResponse("No video file found.", status=400)

        # Get the optional output directory from form
        output_directory = request.POST.get('output_directory', '')

        # Validate/create the output directory if provided
        if output_directory:
            if not os.path.isdir(output_directory):
                try:
                    os.makedirs(output_directory, exist_ok=True)
                except Exception as e:
                    return HttpResponse(f"Invalid output directory: {str(e)}", status=400)
        else:
            # Use a default temp directory if none is given
            output_directory = os.path.join(BASE_DIR, 'temp_uploads')
            os.makedirs(output_directory, exist_ok=True)

        # Save the uploaded file
        temp_video_path = os.path.join(output_directory, 'temp_upload_' + uploaded_file.name)
        with open(temp_video_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # --- Run your DLC processing function here (synchronously) ---
        # For example:
        # csv_file_path = run_dlc_pipeline(temp_video_path, output_directory)

        # TODO: Replace the next line with your actual pipeline call:
        csv_file_path = os.path.join(output_directory, 'output.csv')

        # Ensure the CSV actually exists or handle errors
        if not os.path.exists(csv_file_path):
            return HttpResponse("Error: CSV file not found after processing.", status=500)

        # Return CSV directly as a download
        response = FileResponse(open(csv_file_path, 'rb'), as_attachment=True, filename="results.csv")
        return response

    # If it's not POST, return a simple error response
    return HttpResponse("Invalid request method.", status=400)



@csrf_exempt
def stop_video(request):
    """
    Handles the stop process request.
    Terminates the background process running run_dlc_pipeline.
    """
    global PROCESS, PROCESSING_STATUS
    if request.method == 'POST':
        if PROCESS is not None and PROCESS.is_alive():
            PROCESS.terminate()
            PROCESS.join()
            PROCESS = None
            PROCESSING_STATUS = {"status": "idle", "csv_file": None}
            return JsonResponse({'status': 'Process stopped'})
        else:
            return JsonResponse({'status': 'No active process'})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)


@csrf_exempt
def visualize_csv(request):
    global VIS_PROCESS
    if request.method == 'POST':
        if request.FILES.get('csv_file'):
            # Stop any existing visualization
            if VIS_PROCESS is not None:
                try:
                    VIS_PROCESS.terminate()
                    VIS_PROCESS.wait(timeout=1)
                except:
                    pass
                VIS_PROCESS = None

            csv_file = request.FILES['csv_file']

            # Create a persistent temporary directory if it doesn't exist
            temp_dir = os.path.join(BASE_DIR, 'temp_visualizations')
            os.makedirs(temp_dir, exist_ok=True)

            # Clean up old visualization files
            try:
                for old_file in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, old_file))
                    except:
                        pass
            except:
                pass

            # Save the uploaded file with a unique name
            temp_path = os.path.join(temp_dir, f'vis_{int(time.time())}_{csv_file.name}')

            with open(temp_path, 'wb+') as destination:
                for chunk in csv_file.chunks():
                    destination.write(chunk)

            try:
                # Start the visualization using subprocess
                script_path = os.path.join(BASE_DIR, 'tracking', 'visualization_handler.py')
                VIS_PROCESS = subprocess.Popen([sys.executable, script_path, temp_path])

                # Write initial status
                with open(os.path.join(temp_dir, 'vis_status.json'), 'w') as f:
                    json.dump({'status': 'starting'}, f)

                return JsonResponse({
                    'status': 'Visualization started',
                    'message': 'The visualization window should open shortly. Press ESC in the visualization window or click Stop Visualization to close it.'
                })

            except Exception as e:
                # Only delete the file if there's an error
                try:
                    os.remove(temp_path)
                except:
                    pass
                return JsonResponse({'error': str(e)}, status=500)

        return JsonResponse({'error': 'No CSV file provided'}, status=400)

    elif request.method == 'DELETE':
        # Handle stopping visualization
        if VIS_PROCESS is not None:
            try:
                VIS_PROCESS.terminate()
                VIS_PROCESS.wait(timeout=1)
            except:
                pass
            VIS_PROCESS = None

            # Clean up visualization files
            temp_dir = os.path.join(BASE_DIR, 'temp_visualizations')
            try:
                for old_file in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, old_file))
                    except:
                        pass
            except:
                pass

            return JsonResponse({'status': 'Visualization stopped'})
        return JsonResponse({'status': 'No visualization running'})

    return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def serve_visualization(request, temp_dir, filename):
    """Serve the visualization HTML file"""
    file_path = os.path.join(tempfile.gettempdir(), temp_dir, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        return HttpResponse(content, content_type='text/html')
    raise Http404("Visualization not found")

@csrf_exempt
def run_visualize(request):
    """
    Modified view that creates an MP4 video from the visualization frames and returns it.
    """
    if request.method == 'POST':
        try:
            # Retrieve parameters from the POST data.
            # Here we assume simple form fields; adjust keys to match your form.
            trail_length = int(request.POST.get('trail_length', 10))
            show_trails = request.POST.get('show_trails') == 'on'
            show_trajectory = request.POST.get('show_trajectory') == 'on'
            trajectory_opacity = float(request.POST.get('trajectory_opacity', 0.3))
            show_connections = request.POST.get('show_connections') == 'on'
            show_segmentation = request.POST.get('show_segmentation') == 'on'
            segmentation_opacity = float(request.POST.get('segmentation_opacity', 0.5))
            show_labels = request.POST.get('showLabels') == 'on'
            show_legend = request.POST.get('showLegend') == 'on'

            # Handle the CSV file (and optionally video) uploads.
            csv_file = request.FILES.get('csv_files')
            video_file = request.FILES.get('videos')  # Optional

            if not csv_file:
                return JsonResponse({'status': 'error', 'message': 'No CSV file provided'})

            # Save uploaded CSV file to a temporary location.
            fd, temp_csv_path = tempfile.mkstemp(suffix='.csv')
            with os.fdopen(fd, 'wb') as f:
                for chunk in csv_file.chunks():
                    f.write(chunk)

            temp_video_path = None
            if video_file:
                fd_vid, temp_video_path = tempfile.mkstemp(suffix='.mp4')
                with os.fdopen(fd_vid, 'wb') as f:
                    for chunk in video_file.chunks():
                        f.write(chunk)

            # Define the output video file path.
            temp_dir = os.path.join(BASE_DIR, 'temp_visualizations')
            os.makedirs(temp_dir, exist_ok=True)
            output_video = os.path.join(temp_dir, f'visualization_{int(time.time())}.mp4')

            # Create an instance of your visualizer.
            from tracking.visualization import RodentVisualizerCV
            visualizer = RodentVisualizerCV(
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

            # Write the animation to an MP4 using the new method.
            visualizer.write_animation_to_video(output_video, fps=30)

            # Clean up temporary CSV and video files.
            try:
                os.remove(temp_csv_path)
                if temp_video_path:
                    os.remove(temp_video_path)
            except Exception as e:
                print("Cleanup error:", e)

            # Return the video file to the client.
            #return FileResponse(open(output_video, 'rb'), content_type='video/mp4')
            # After generating and closing the video file, use the following:
            response = FileResponse(
                open(output_video, 'rb'),
                content_type='video/mp4',
                as_attachment=True,
                filename=os.path.basename(output_video)
            )
            return response

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)