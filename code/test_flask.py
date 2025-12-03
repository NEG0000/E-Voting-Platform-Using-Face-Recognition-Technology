from flask import Flask

app = Flask(__name__)

@app.before_first_request
def run_once():
    print("Before first request")

@app.route('/')
def hello():
    return "Hello, Flask!"

if __name__ == '__main__':
    app.run(debug=True)
