import pandas as pd
import psycopg2
import torch
from transformers import BertTokenizer, BertModel

# Load BERT model
tokenizer = BertTokenizer.from_pretrained("/path/to/local/bert-base-uncased")
model = BertModel.from_pretrained("/path/to/local/bert-base-uncased")
model.eval()

def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']

        # Mean Pooling
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_embeddings = last_hidden_state * mask
        summed = torch.sum(masked_embeddings, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts

        return mean_pooled.squeeze().tolist()

def insert_csv_to_pgvector(csv_path: str):
    df = pd.read_csv(csv_path)

    conn = psycopg2.connect(
        host="localhost",
        dbname="your_db",
        user="your_user",
        password="your_password",
        port=5432
    )
    cur = conn.cursor()

    for idx, row in df.iterrows():
        text = row["entry_text"]
        failure = row["failure_type"]
        resolution = row["resolution"]
        vector = get_embedding(text)

        cur.execute("""
            INSERT INTO api_knowledge_base (entry_text, embedding, failure_type, resolution)
            VALUES (%s, %s, %s, %s)
        """, (text, vector, failure, resolution))

    conn.commit()
    cur.close()
    conn.close()
    print("Data inserted from CSV into PostgreSQL.")

if __name__ == "__main__":
    insert_csv_to_pgvector("data/dataset.csv")
