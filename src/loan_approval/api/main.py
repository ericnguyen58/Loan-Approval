"""FastAPI entrypoint."""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loan_approval.api.routes import router

app = FastAPI(title="Let Me Have a Loan", version="0.1.0")
app.include_router(router)


@app.exception_handler(RequestValidationError)
def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Flatten Pydantic's nested error dump into one message per bad field."""
    errors = [
        {"field": ".".join(str(p) for p in err["loc"][1:]), "issue": err["msg"]}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def run() -> None:
    uvicorn.run("loan_approval.api.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    run()
