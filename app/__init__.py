"""External-validation web app.

Thin FastAPI wrapper over :mod:`src.inference` that lets a user upload an
unseen Morris gas-pipeline ARFF/CSV and score it against a saved model
artifact. Localhost-only — no auth, no rate limiting, no job queue.

Run with::

    uvicorn app.main:app --reload

and open http://localhost:8000/ in a browser.
"""
