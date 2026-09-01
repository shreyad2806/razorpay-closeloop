from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analyze import AnalyzeRequest, AnalyzeResponse, AnalyzeService
from app.api.explain import ExplainRequest, ExplainResponse, ExplainService

app = FastAPI(
    title="Razorpay CloseLoop"
)

# Singleton services
_explain_service = ExplainService()
_analyze_service = AnalyzeService()


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/explain", response_model=ExplainResponse)
async def explain_exception(request: ExplainRequest):
    """Provide a human-readable explanation of a financial exception.

    Accepts an exception ID and returns a structured explanation
    including summary, evidence, uncertainty, and LLM provider status.

    Does NOT accept arbitrary financial truth values as inputs.
    """
    return await _explain_service.explain(request)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_exception(request: AnalyzeRequest):
    """Provide a complete AI-assisted investigation summary.

    Combines all existing results: reconciliation, evidence,
    classification, candidates, guardrails, and LLM explanation.

    Does NOT allow the LLM to calculate financial values,
    choose resolutions, or override guardrails.
    """
    return await _analyze_service.analyze(request)
