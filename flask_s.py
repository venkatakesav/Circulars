from flask import Flask, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return 'Server is running'

@app.route('/<path:filename>')
def serve_file(filename):
    root_dir = "/Users/venkatakesavvenna/Research/Annotation_Pipeline"  # Replace with the path to your files
    return send_from_directory(root_dir, filename)

if __name__ == '__main__':
    app.run(port=8081, debug=True)
