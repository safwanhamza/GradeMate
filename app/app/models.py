from django.db import models
from django.contrib.auth.models import User

class ExamUpload(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.file.name}"


class GradingSettings(models.Model):
    CONTEXT_CHOICES = [
        ('textbook', 'Textbook'),
        ('both', 'Both Textbook & AI'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    textbook = models.FileField(upload_to='textbooks/')
    key = models.FileField(upload_to='keys/')
    similarity = models.FloatField()
    creativity = models.FloatField()
    manual_text = models.TextField(blank=True)
    context_choice = models.CharField(max_length=20, choices=CONTEXT_CHOICES)

    def __str__(self):
        return f"Settings for {self.user.username}"


class ExtractedAnswer(models.Model):
    exam = models.ForeignKey(ExamUpload, on_delete=models.CASCADE)
    question_index = models.IntegerField()
    text = models.TextField()

    def __str__(self):
        return f"Q{self.question_index} - {self.exam.user.username}"


class GradedAnswer(models.Model):
    extracted = models.ForeignKey(ExtractedAnswer, on_delete=models.CASCADE)
    score = models.FloatField()
    feedback = models.TextField()

    def __str__(self):
        return f"Grade for {self.extracted.exam.user.username} - Q{self.extracted.question_index}"
