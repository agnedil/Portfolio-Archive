from flask import Flask
from config import Config

myapp = Flask(__name__, instance_path='/home/.../app/instance/')    # indicate full path here
# ,static_url_path="/home/..../app/static/"                         # indicate full path here
myapp.config.from_object(Config)

from app import routes
