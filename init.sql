-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table to store clinical text chunks & vector embeddings
CREATE TABLE IF NOT EXISTS clinical_embeddings (
    id BIGSERIAL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    source_table VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384), -- 384 dimensions for all-MiniLM-L6-v2 embeddings
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fast HNSW index for cosine distance similarity searches
CREATE INDEX IF NOT EXISTS idx_clinical_vector_hnsw 
ON clinical_embeddings 
USING hnsw (embedding vector_cosine_ops);