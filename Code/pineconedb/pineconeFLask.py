from flask import Flask, request, jsonify
from flask_cors import CORS

# 👇 Import agent_executor from your pcVB.py file
from flpcVB import agent_executor

app = Flask(__name__)
CORS(app)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "Query is required."}), 400

    try:
        response = agent_executor.invoke({"input": query})
        return jsonify({"answer": response["output"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False) 
