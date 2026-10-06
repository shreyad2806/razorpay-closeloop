"""API Routes package for Razorpay CloseLoop."""

from app.api.routes.auth import router as auth_router
from app.api.routes.batches import router as batches_router
from app.api.routes.exceptions import router as exceptions_router
from app.api.routes.intelligence import router as intelligence_router
from app.api.routes.learning import router as learning_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.models import router as models_router

all_routers = [
    auth_router,
    batches_router,
    exceptions_router,
    intelligence_router,
    learning_router,
    metrics_router,
    models_router,
]

