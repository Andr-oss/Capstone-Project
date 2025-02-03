# tracking/views.py
from django.shortcuts import render, redirect


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        error = None

        if username != 'admin':
            error = "Invalid username"
        elif password != 'admin':
            error = "Invalid password"

        if error:
            # Return the error message and pre-fill the username if available.
            return render(request, 'tracking/login.html', {'error': error, 'username': username})

        # Mark the user as authenticated in the session
        request.session['is_authenticated'] = True
        return redirect('dashboard')

    return render(request, 'tracking/login.html')


def dashboard_view(request):
    # Check if the user is authenticated
    if not request.session.get('is_authenticated'):
        return redirect('login')

    return render(request, 'tracking/dashboard.html')

def new_videos_view(request):
    if not request.session.get('is_authenticated'):
        return redirect('login')

    # You can add more backend logic here as needed later.
    return render(request, 'tracking/new_videos.html')