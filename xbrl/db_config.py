"""
Database connection configuration.
Reads from environment variables with safe local defaults.
"""
import os

DB_CONFIG = {
    "host":     os.getenv("PGHOST",     "localhost"),
    "port":     int(os.getenv("PGPORT", "5432")),
    "dbname":   os.getenv("PGDATABASE", "ma_oracle"),
    "user":     os.getenv("PGUSER",     "postgres"),
    "password": os.getenv("PGPASSWORD", ""),
}