from model import get_context_from_pg, generate_email_response

if __name__ == "__main__":
    user_log = "Getting 500 error on /search with high latency from EU users"

    print("🔍 Query:", user_log)

    # Retrieve similar failures from the pgvector knowledge base
    context = get_context_from_pg(user_log)
    print("\n Retrieved Context:\n", context)

    # Generate email response using TinyLLaMA
    email = generate_email_response(user_log, context)
    print("\n Suggested Email:\n", email)