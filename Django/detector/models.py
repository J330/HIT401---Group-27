from django.conf import settings
from django.db import models


class ScanHistory(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='scan_history',
        null=True,
        blank=True,
    )
    session_key = models.CharField(max_length=40, blank=True, db_index=True)
    image = models.ImageField(upload_to='scan_history/', blank=True, null=True)
    label = models.CharField(max_length=64)
    confidence = models.FloatField(default=0.0)
    result = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user} - {self.label} - {self.created_at:%Y-%m-%d %H:%M}'
