from fastapi import FastAPI

app = FastAPI(
    title="Razorpay CloseLoop"
)

@app.get("/health")
def health():
    return {
        "status": "ok"
    }
