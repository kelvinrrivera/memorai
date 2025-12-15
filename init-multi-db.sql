CREATE DATABASE cognee_db;

-- Asignar permisos
GRANT ALL PRIVILEGES ON DATABASE cognee_db TO postgres;


-- Conectar a cognee_db e instalar vector
\c cognee_db
CREATE EXTENSION IF NOT EXISTS vector;