from pymongo import MongoClient
from app.core.config import settings


try:
    client = MongoClient(
        settings.MONGODB_URI,
        tls=True,
        tlsAllowInvalidCertificates=True,
    )
    client.admin.command('ping')
    db = client[settings.DB_NAME]
except Exception as e:
    # In server startup, we let exceptions surface; FastAPI will show error
    raise
