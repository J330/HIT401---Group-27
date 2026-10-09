# Backend setup

manage.py and backend/backend/{__init__.py, asgi.py, wsgi.py, admin.py} and
detector/{__init__.py, apps.py, migrations/} are NOT included in this zip --
Django generates them for you. Run this first, from inside backend/:

```
django-admin startproject backend .
python manage.py startapp detector
```

That will create manage.py and the files above. It will ALSO create fresh
copies of backend/backend/settings.py, backend/backend/urls.py, and (once
you run startapp) detector/ as an empty app package.

**Immediately after running those two commands, overwrite the generated**
**settings.py and urls.py with the ones already sitting in this folder**
(backend/backend/settings.py and backend/backend/urls.py) -- they're
pre-written with DRF, CORS, WhiteNoise, and the ACTIVE_MODEL setting
already configured. Do the same for detector/ -- copy the models.py,
serializers.py, admin.py, api_docs.py, inference.py, views.py, and
urls.py already in this folder into the detector/ app Django just created
for you (they'll overwrite the empty ones startapp generates).

Then:

```
python manage.py makemigrations detector
python manage.py migrate
python manage.py createsuperuser   # optional, for /admin/
python manage.py runserver
```

See the top-level README.md for the full order of operations.
