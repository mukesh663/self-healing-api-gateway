from flask import Flask, request, jsonify
from model import get_context_from_pg, generate_email_response

app = Flask(__name__)

@app.route("/generate-response", methods=["POST"])
def generate_response():
    data = request.get_json()
    user_log = data.get("user_log")

    if not user_log:
        return jsonify({"error": "Missing 'user_log' in request"}), 400

    try:
        context = get_context_from_pg(user_log)
        email = generate_email_response(user_log, context)

        return jsonify({
            "context": context,
            "email_response": email
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
