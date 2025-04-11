from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from segmenter_api.domain.model.errors import BaseApiError


def create_api() -> FastAPI:
    app = FastAPI(openapi_url="/openapi.json", docs_url="/docs", redoc_url="/redoc")

    setup_router(app)
    setup_exception_handler(app)
    setup_cors(app)

    return app


def setup_router(app: FastAPI) -> None:
    from segmenter_api.app.router import router

    app.include_router(router)

    @app.get("/")
    async def index(request: Request) -> JSONResponse:
        return JSONResponse(content={"status": "successful"}, status_code=200)

    @app.get("/status")
    async def status() -> JSONResponse:
        return JSONResponse(content={"status": "successful"}, status_code=200)


def setup_exception_handler(app: FastAPI) -> None:
    @app.exception_handler(BaseApiError)
    async def api_error_handler(request: Request, err: BaseApiError) -> JSONResponse:
        return JSONResponse(status_code=err.status_code, content={"message": err.msg})

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, err: Exception
    ) -> JSONResponse:
        return JSONResponse(status_code=500, content={"message": str(err)})


def setup_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


api = create_api()
