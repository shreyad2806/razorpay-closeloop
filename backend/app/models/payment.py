from sqlalchemy import Column, String, Float
from app.database.database import Base

class Payment(Base):
    __tablename__ = "payments"

    id = Column(String, primary_key=True)
    merchant_id = Column(String)
    amount = Column(Float)
