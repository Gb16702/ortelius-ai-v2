from fastapi import Request, FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logfire

class ErrorHandler:
    def __init__(self, app: FastAPI):
        self.app = app
        self.register_exception_handlers()

    def register_exception_handlers(self):
        self.app.add_exception_handler(
            RequestValidationError,
            self.validation_exception_handler
        )

    async def validation_exception_handler(self, request: Request, exc: RequestValidationError):
        errors = []

        for error in exc.errors():
            logfire.error("Validation error", error=error)

            field = error.get("loc", ["unknown"])[-1] if error.get("loc") else "unknown"
            error_type = error.get("type", "unknown")
            message = error.get("msg", "Unknown error")

            errors.append({
                "field": str(field),
                "type": error_type,
                "message": message
            })

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"errors": errors}
        )