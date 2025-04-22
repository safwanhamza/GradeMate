from django.contrib import admin
from .models import ExamUpload, GradingSettings, ExtractedAnswer, GradedAnswer

admin.site.register(ExamUpload)
admin.site.register(GradingSettings)
admin.site.register(ExtractedAnswer)
admin.site.register(GradedAnswer)
