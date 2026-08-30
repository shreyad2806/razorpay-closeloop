from app.database.database import Base, engine
from app.models.payment import Payment
from app.models.settlement import Settlement
from app.models.exception import FinancialException

Base.metadata.create_all(bind=engine)

print("Tables created")
