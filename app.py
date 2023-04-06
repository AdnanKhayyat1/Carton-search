from flask import Flask, request, Response, jsonify
from engine import SearchEngine
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)


@app.route('/search/', methods=['GET'])
def search():
    prompt = request.args.get('prompt')
    if prompt:
        eng = SearchEngine(openai_key=os.getenv('OPENAI_API_KEY'))
        df = eng.parse_data(uri=os.getenv('ATLAS_URI'))
        document_embeddings = eng.compute_doc_embeddings(df)
        answerData = eng.answer_query_with_context(
            prompt, df, document_embeddings)
        response = {'answer': answerData}
        return response, 200, {'Access-Control-Allow-Origin': '*'}
    else:
        resp = Response({'answer', 'No prompt specified!'}, status=404)
        return resp


@app.route('/')
def hello_world():
    return 'Hello, World!'
