import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes
from app.config import settings
from app.db import init_db
from app.logger import get_logger, setup_logging

setup_logging()
log = get_logger("main")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        log.info("REQUEST  %s %s", request.method, request.url.path)
        try:
            response = await call_next(request)
        except Exception as exc:
            log.error("UNHANDLED EXCEPTION %s %s — %s", request.method, request.url.path, exc, exc_info=True)
            raise
        elapsed = (time.perf_counter() - start) * 1000
        level = log.warning if response.status_code >= 400 else log.info
        level("RESPONSE %s %s → %d (%.1fms)", request.method, request.url.path, response.status_code, elapsed)
        return response

    app.include_router(routes.router)

    @app.on_event("startup")
    def on_startup():
        log.info("Starting %s v%s | provider=%s", settings.app_name, settings.app_version, settings.llm_provider)
        try:
            init_db()
            log.info("Database initialised successfully")
        except Exception as exc:
            log.error("Database initialisation failed: %s", exc, exc_info=True)
            raise

    @app.on_event("shutdown")
    def on_shutdown():
        log.info("Application shutting down")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
