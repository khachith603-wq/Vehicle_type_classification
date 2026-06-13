from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('classify/', views.classify_view, name='classify'),
    path('history/', views.history_view, name='history'),
    path('about/', views.about_view, name='about'),
    path('help/', views.help_view, name='help'),
]
