import csv
import sys, pathlib, os, multiprocessing, time, subprocess
import tempfile
import json
import shutil
import threading
import uuid

import deeplabcut
from pathlib import Path
import glob
import os

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, FileResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from tracking.post_process import postprocess_csv



# Set BASE_DIR and update sys.path to import backend modules outside the frontend folder
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR.parent))

# Import your backend function from dlc_runner
# import dlc_runner

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


# Global dictionary to track processing tasks
processing_tasks = {}

@csrf_exempt
def process_video_background(task_id, video_path, config_path, temp_dir):
    """Function to run video processing in background"""
    try:
        processing_tasks[task_id]['status'] = 'processing'

        # Run DeepLabCut analysis
        deeplabcut.analyze_videos(
            config=config_path,
            videos=[video_path],
            save_as_csv=True,
            destfolder=temp_dir,
            auto_track=False
        )

        # Find the generated CSV
        video_basename = Path(video_path).stem
        csv_files = sorted(glob.glob(os.path.join(temp_dir, f"*{video_basename}*DLC*.csv")),
                           key=os.path.getmtime, reverse=True)

        if not csv_files:
            # Try looking for any DLC csv
            csv_files = sorted(glob.glob(os.path.join(temp_dir, "*DLC*.csv")),
                               key=os.path.getmtime, reverse=True)

        if not csv_files:
            processing_tasks[task_id]['status'] = 'error'
            processing_tasks[task_id]['message'] = 'No output CSV generated'
            return

        latest_csv = csv_files[0]

        # Apply post-processing
        processing_tasks[task_id]['status'] = 'post-processing'

        # Call post_process and get the processed content
        processed_csv_path = os.path.join(temp_dir, f"processed_{os.path.basename(latest_csv)}")
        print("Post-processing CSV...")
        postprocess_csv(latest_csv, processed_csv_path)
        print("processed_csv_path is ready")


        # Store the csv path
        with open(processed_csv_path, 'rb') as f:
            processing_tasks[task_id]['csv_content'] = f.read()
            processing_tasks[task_id]['csv_filename'] = os.path.basename(processed_csv_path)
            processing_tasks[task_id]['status'] = 'complete'

    except Exception as e:
        processing_tasks[task_id]['status'] = 'error'
        processing_tasks[task_id]['message'] = str(e)

@csrf_exempt
def process_video(request):
    if request.method == 'POST':
        # Check if any videos were uploaded
        if 'videos' not in request.FILES:
            return JsonResponse({'status': 'error', 'message': 'No videos uploaded'}, status=400)

        uploaded_files = request.FILES.getlist('videos')
        if not uploaded_files:
            return JsonResponse({'status': 'error', 'message': 'No videos selected'}, status=400)

        # Get the model option
        model_option = request.POST.get('model_option', 'deeplabcut')

        # Only proceed with DeepLabCut model
        if model_option == 'deeplabcut':
            try:
                # Create a directory for processing (you'll need to manage cleanup)
                temp_dir = os.path.join('media', 'temp', str(uuid.uuid4()))
                os.makedirs(temp_dir, exist_ok=True)

                # Process the first video
                video_file = uploaded_files[0]

                # Save the uploaded video
                video_path = os.path.join(temp_dir, video_file.name)
                with open(video_path, 'wb+') as destination:
                    for chunk in video_file.chunks():
                        destination.write(chunk)

                # Generate a task ID
                task_id = str(uuid.uuid4())

                # Store task info
                processing_tasks[task_id] = {
                    'status': 'starting',
                    'video_path': video_path,
                    'temp_dir': temp_dir,
                    'start_time': time.time()
                }

                # Start background processing
                config_path = os.path.join(BASE_DIR.parent, 'Capstone-App', 'DLC Trained Model', 'config.yaml')
                thread = threading.Thread(
                    target=process_video_background,
                    args=(task_id, video_path, config_path, temp_dir)
                )
                thread.daemon = True
                thread.start()

                # Return task ID immediately
                return JsonResponse({'status': 'started', 'task_id': task_id})

            except Exception as e:
                return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
        else:
            return JsonResponse({'status': 'error', 'message': 'Selected model is not available'}, status=400)

    # If not a POST request
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

@csrf_exempt
def check_task_status(request):
    """View to check the status of video processing"""
    task_id = request.GET.get('task_id')

    if not task_id or task_id not in processing_tasks:
        return JsonResponse({'status': 'error', 'message': 'Invalid task ID'}, status=400)

    task = processing_tasks[task_id]

    # Create response based on current status
    response = {'status': task['status']}

    if task['status'] == 'error':
        response['message'] = task.get('message', 'Unknown error')
    elif task['status'] == 'complete':
        response['message'] = 'Processing complete'
        response['filename'] = task.get('csv_filename', 'output.csv')
    else:
        # Calculate time elapsed
        elapsed = time.time() - task['start_time']
        response['elapsed_seconds'] = int(elapsed)
        response['message'] = f'Processing for {int(elapsed)} seconds...'

    return JsonResponse(response)

@csrf_exempt
def download_result(request):
    """View to download the processed result"""
    task_id = request.GET.get('task_id')

    if not task_id or task_id not in processing_tasks:
        return HttpResponse("Invalid task ID", status=400)

    task = processing_tasks[task_id]

    if task['status'] != 'complete':
        return HttpResponse("Processing not complete", status=400)

    # Prepare response
    response = HttpResponse(task['csv_content'], content_type='text/csv')
    video_name = Path(task['video_path']).stem
    response['Content-Disposition'] = f'attachment; filename="{video_name}_processed.csv"'

    # Delete the temp directory
    temp_dir = task.get('temp_dir')
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Delete the task from the dictionary
    del processing_tasks[task_id]

    return response


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
            from visualization import RodentVisualizerCV
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