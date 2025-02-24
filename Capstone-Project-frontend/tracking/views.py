from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, authenticate
import csv
from django.http import HttpResponse

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        error = None

        # Authenticate using Django's built-in authentication
        from django.contrib.auth import authenticate
        user = authenticate(request, username=username, password=password)
        if user is None:
            error = "Invalid username or password"
            return render(request, 'tracking/login.html', {'error': error, 'username': username})

        # Log the user in
        from django.contrib.auth import login
        login(request, user)
        return redirect('dashboard')

    return render(request, 'tracking/login.html')


def dashboard_view(request):
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'tracking/dashboard.html')


def new_videos_view(request):
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'tracking/new_videos.html')


def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()  # This saves the new user to the database (SQLite by default)
            # Optionally, log the user in immediately after registration
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            new_user = authenticate(username=username, password=raw_password)
            if new_user is not None:
                login(request, new_user)
                return redirect('dashboard')
            # Alternatively, redirect to the login page:
            # return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'tracking/register.html', {'form': form})


def download_csv(request):
    # Create a HttpResponse object with CSV content
    response = HttpResponse(content_type='text/csv')
    # Set the Content-Disposition header to prompt download with a filename
    response['Content-Disposition'] = 'attachment; filename="tracking_data.csv"'

    writer = csv.writer(response)
    # Write the header row with the specified titles
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
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'tracking/livestream.html')
