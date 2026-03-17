# RQP Chatbot Backend (FastAPI)

This is a minimal FastAPI backend for the RQP Chatbot.

## Setup

```bash
# (optional) create and activate a virtualenv first
pip install -r requirements.txt
```

## Run locally

```bash
# from the rqp-chatbot-backend folder
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# or
python -m app.main
```

The API will be available at:

- OpenAPI docs: http://localhost:8000/docs
- ReDoc docs: http://localhost:8000/redoc
- Health check: http://localhost:8000/health
- Liveness: http://localhost:8000/live
- Readiness: http://localhost:8000/ready

## Notes

- CORS is configured for `http://localhost:4200` and `http://localhost:4300` to work with the Angular frontends.
- Adjust allowed origins and ports in `app/main.py` if your UI runs elsewhere.
