import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

postgres_user = os.environ.get("POSTGRES_USER")
postgres_password = os.environ.get("POSTGRES_PASSWORD")
postgres_host = os.environ.get("POSTGRES_HOST")
postgres_port = os.environ.get("POSTGRES_PORT", "5432")
postgres_db = os.environ.get("POSTGRES_DB")

if postgres_user and postgres_password and postgres_host and postgres_db:
    DATABASE_URL = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
else:
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./trading_system.db")

try:
    engine = create_engine(DATABASE_URL, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info(f"Database engine created for {DATABASE_URL}")
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    raise

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
