from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from DocumentManager.forms import RegistrationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .utils import get_pdf_stats  # To show PDF stats in index.html

def Home(request):
    filter_type = request.GET.get('filter', 'all')  # Optional: allows ?filter=today, etc.
    stats_context = get_pdf_stats(filter_type)
    stats_context['username'] = request.user.username  # Add username to context
    return render(request, 'index.html', stats_context)

def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            return redirect('registration/login')
    else:
        form = RegistrationForm()

    return render(request, 'registration/register.html', {'form': form})

def user_login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('/')  
        else:
            return render(request, 'registration/login.html', {'error': 'Invalid Credentials'})
    return render(request, 'registration/login.html')

def user_logout(request):
    logout(request)
    return redirect('/')

@login_required(login_url='/login/')
def pipeline(request):
    return render(request,'pipeline.html')

def page_not_found(request):
    return render(request,'page_not_found.html')

















