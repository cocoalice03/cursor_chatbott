from django.contrib import admin
from django.urls import path
from users.views import check_quota

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/check-quota/', check_quota, name='check_quota'),  # ✅ endpoint clé
]
