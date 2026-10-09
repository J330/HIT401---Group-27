# Django + Original Frontend — One Server

Open PowerShell in this backend folder (where manage.py is located):

```powershell
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Visit http://127.0.0.1:8000/ (Home), or http://127.0.0.1:8000/upload.html (Upload).

No `python -m http.server 8080` is needed. All four original frontend pages and their CSS/JS are now hosted by Django. The scan sends POST /api/predict/ on the same origin.

Trained AI checkpoints are NOT in this ZIP. The prediction backend expects files in `../ml/models/` relative to the backend directory. Install the Python dependencies and supply trained files for real scans.

For production, run collectstatic and configure your static-file server as usual.
