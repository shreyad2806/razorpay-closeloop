from sqlalchemy import Column, String, Float
from app.database.database import Base

class FinancialException(Base):
    __tablename__ = "exceptions"

    id = Column(String, primary_key=True)
    payment_id = Column(String)
    expected_amount = Column(Float)
    actual_amount = Column(Float)
    difference = Column(Float)
    exception_type = Column(String)
    status = Column(String)
