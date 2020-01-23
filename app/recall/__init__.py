from flask import Flask
app = Flask(__name__)
app.static_folder = 'static'
from recall import views