# 🔧 Configuration de la base de données PostgreSQL
DB_CONFIG = {
    "dbname": "electric_db",
    "user": "postgres",
    "password": "aoufnihed",
    "host": "localhost",
    "port": "5432"
}

# ⚡ Seuils de surveillance
THRESHOLDS = {
    "voltage_min": 210,
    "voltage_max": 230,
    "harmonic_limit": 5,
    "frequency_nominal": 50,
    "current_max": 100
}
