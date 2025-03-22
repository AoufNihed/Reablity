from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ✅ Database Configuration with Your Credentials
DATABASE_URL = "postgresql://postgres:aoufnihed@localhost/electric_db"

# ✅ Create Engine & Session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# ✅ Base Model
Base = declarative_base()

# ✅ Define Data Table
class ElectricData(Base):
    __tablename__ = "electric_data"

    id = Column(Integer, primary_key=True, index=True)
    voltage = Column(Float, nullable=False)
    current = Column(Float, nullable=False)
    harmonics = Column(Float, nullable=True)
    load = Column(Float, nullable=False)
    status = Column(String, nullable=False)

# ✅ Create Tables
def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("✅ Database initialized successfully!")
