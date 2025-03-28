from django.urls import path
from .views import check_quota

urlpatterns = [
    path('check-quota/', check_quota, name='check_quota'),
]
