"""
ASGI config for app project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""

import os
import sys, io, logging

# Re-wrap stdout / stderr with UTF-8 so *any* print / logger->StreamHandler works
if hasattr(sys.stdout, "reconfigure"):          # Python ≥3.7
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
else:                                           # very old Pythons
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                                  encoding="utf-8",
                                  errors="backslashreplace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer,
                                  encoding="utf-8",
                                  errors="backslashreplace")
    
from django.core.asgi import get_asgi_application
import os
os.environ["ATLAS_API_KEY"] = "nk-rjqDQKJtuoRaTcocIvaSJr6g5JItcyLvNJR4O7h153o"
os.environ["TRANSFORMERS_NO_TF"] = "1"      # blocks TF imports globally

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.settings')

application = get_asgi_application()
