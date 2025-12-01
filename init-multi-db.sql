-- Crear Bases de Datos separadas
CREATE DATABASE mem0_db;
CREATE DATABASE cognee_db;

-- Asignar permisos
GRANT ALL PRIVILEGES ON DATABASE mem0_db TO postgres;
GRANT ALL PRIVILEGES ON DATABASE cognee_db TO postgres;

-- Conectar a mem0_db e instalar vector
\c mem0_db
CREATE EXTENSION IF NOT EXISTS vector;

-- Conectar a cognee_db e instalar vector
\c cognee_db
CREATE EXTENSION IF NOT EXISTS vector;