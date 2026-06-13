from django.db import models
from django.contrib.auth.models import User


class Prediction(models.Model):
    user         = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    image        = models.ImageField(upload_to='predictions/')
    label        = models.CharField(max_length=200)
    confidence   = models.FloatField()
    vehicle_type = models.CharField(max_length=100, default='Unknown')
    created_at   = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} – {self.vehicle_type} | {self.label} ({self.confidence:.1f}%)"

    class Meta:
        ordering = ['-created_at']
