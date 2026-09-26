Initial Django Framework Setup

## 23/08/2026 Revision:

### Status:

Approved

### Context

The Django framework needed to be set up before development could begin. A virtual environment was created, Django was installed, and the development server was tested successfully.

### Alternative Solutions Considered

1. Install Django globally

   - Pros

     - Simple setup.

   - Cons

     - Can cause package conflicts between projects.

2. Use a virtual environment

   - Pros

     - Keeps project packages isolated.
     - Easier dependency management.

   - Cons

     - Must be activated before development.

### Solution Decided Upon:

A virtual environment was created and Django was installed inside it. A Django project was then created and tested using the development server.

### Consequences:

The project now has a working Django framework and an isolated environment for future packages and development.

### Code Reference:

```python
python -m venv .venv
.venv\Scripts\activate
python -m pip install django
```

```python
django-admin startproject projectname .
python manage.py runserver
```
