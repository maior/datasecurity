from django.urls import include, path

from .views import (
    MaliciousApiView,
)

urlpatterns = [
    path('api', MaliciousApiView.as_view()),
]