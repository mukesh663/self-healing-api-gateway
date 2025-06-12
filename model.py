import torch
import psycopg2
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer, BertModel

# ─── Load Local TinyLLaMA for Generation ────────────────────────────
tinyllama_path = "/path/to/tinyllama"

tokenizer = AutoTokenizer.from_pretrained(tinyllama_path)
model = AutoModelForCausalLM.from_pretrained(
    tinyllama_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ─── Load Local BERT for Embedding ──────────────────────────────────
bert_path = "/path/to/local/bert-base-uncased"

bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
bert_model = BertModel.from_pretrained(bert_path)
bert_model.eval()

def get_embedding(text):
    with torch.no_grad():
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = bert_model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']

        # Mean Pooling
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_embeddings = last_hidden_state * mask
        summed = torch.sum(masked_embeddings, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts

        return mean_pooled.squeeze().tolist()

# ─── RAG Context from pgvector ──────────────────────────────────────
def get_context_from_pg(query: str, top_k: int = 3):
    vec = get_embedding(query)

    conn = psycopg2.connect(
        host="localhost", dbname="your_db",
        user="your_user", password="your_password", port=5432
    )
    cur = conn.cursor()

    cur.execute("""
        SELECT entry_text, failure_type, resolution
        FROM api_knowledge_base
        ORDER BY embedding <#> %s
        LIMIT %s
    """, (vec, top_k))

    results = cur.fetchall()
    cur.close()
    conn.close()

    return "\n".join([
        f"- [{ft}] {et}\n  Resolution: {res}"
        for et, ft, res in results
    ])

# ─── Generate Email Response ─────────────────────────
def generate_email_response(user_query, retrieved_context):
    prompt = f"""
        You are an API healing assistant. Based on past issues and their resolutions, suggest a fix.

        Problem: {user_query}

        Similar Past Failures:
        {retrieved_context}

        Compose an email to the dev team with the resolution. Be concise, technical, and include any recommended steps or next actions.
        """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)