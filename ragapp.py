from flask import Flask, render_template, request
from rag import rag  # Import the RAG function from the rag module
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from the .env file

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        query = request.form.get("query")
        if query:
            answer = rag(query)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
