from sqlalchemy import Column, String, Integer
from app.database.database import Base

class Settlement(Base):
    __tablename__ = "settlements"

    id = Column(String, primary_key=True)
    payment_id = Column(String)
    amount = Column(Integer)  # Financial amount in paise (integer)
