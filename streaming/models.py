from django.conf import settings
from django.db import models
from django.conf import settings

class Videoo(models.Model):
    user=models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=100)
    like =models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        blank=True,
        related_name='like'
    )

    def count_likes(self):
        return self.like.count()

    def __str__(self):
        return self.title