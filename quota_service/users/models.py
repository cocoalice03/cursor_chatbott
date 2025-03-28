from django.db import models

class UserQuota(models.Model):
    email = models.EmailField(unique=True)
    date = models.DateField(auto_now=True)
    question_count = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.email} - {self.date} - {self.question_count} questions"
