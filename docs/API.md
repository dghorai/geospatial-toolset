API
=======
API -> Application Programming Interface

It is a system to interact between different technology stack. 

Example: 

One app/system is developed using Java language and another app/system is developed using Python language, to use some features/functions of one app to another app we need to use some mediator that will interact between these two different app or heterogenious system or sometimes for homogenious system. This is done by API. API is more of a protocol where some kind of instruction is provided to do the job.

Requirements
==============
1) [Postman](https://www.postman.com/downloads/) -> either create an a/c or just simply use Lighweight API client for practice/test purpose.
2) IDE (PyCharm/VSCode)
3) Flask
4) Database (PostgreSQL) (optional)


Create API
=============
-> Create some function using IDE

Example:
```
from flask import Flask, request, jsonify

app = Flask(__name__)  # create a flask object

@app.route('/xyz', methods=['GET', 'POST'])  # reach out to the function below using route function in flask
def test(a, b):
	if (request.method == 'POST'):
		a = request.json['num1']
		b = request.json['num2']
		result = a + b
		return jsonify(result)

if __name__ == '__main__':
	app.run()

# start server/run this flask app; it will generate an URL
# get the API url: http://127.0.0.1:5000/xyz (example)

# GET and POST method both using to send data
# Search on Google is example of GET method (resultant data will be visible)
# Login gmail a/c is example of POST method (resultant data will not be visible)
```

Test API in Postman
=====================
1) Select method (GET/POST)
2) Enter URL next to the method (say, http://127.0.0.1:5000/xyz)
3) Fill Json `Body` in Postman as per the function parameters in the above (ex., num1, num2)
    - select 'Body' -> select 'raw' -> change text to JSIN insert some json data:
    - example:
        {
        "num1": 3,
        "num2": 4
        }
4) Click on 'Send' button