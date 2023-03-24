import os
import sys
from flask import Flask
sys.path.append("../../models/src")
# from models import ...

app = Flask(__name__)

@app.route("/")
def index():
    return "/ Endpoint"


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--debug", type=bool, default=False)

    args = parser.parse_args()

    app.run(port=args.port, host=args.host, debug=args.debug)