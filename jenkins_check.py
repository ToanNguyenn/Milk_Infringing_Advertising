import numpy
from pymongo import MongoClient
import time
from flask import Flask, request

app = Flask(__name__)
# client = MongoClient("localhost", 00000)

@app.route('/')
def hello_world():
    return f"Update_{time.time()}, Hello world"
if __name__ == '__main__':
    print("App run!!!")
    app.run(debug=True)