-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the knowledge base table
CREATE TABLE IF NOT EXISTS api_knowledge_base (
    id SERIAL PRIMARY KEY,
    entry_text TEXT NOT NULL,
    embedding vector(384),  -- Assuming MiniLM-L6 (384 dims)
    failure_type TEXT,
    resolution TEXT
);

-- Create a vector index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_embedding_vector
ON api_knowledge_base USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
