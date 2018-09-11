from flask import Flask
from config import Config

myapp = Flask(__name__, instance_path='/home/andrew/Documents/2_UIUC/CS598_Data_Mining_Capstone/task7_myapp_isIn_myFlaskVirtualEnv/myflask/myapp/app/instance/')
# ,static_url_path="/home/andrew/Documents/2_UIUC/CS598_Data_Mining_Capstone/task7_myapp_isIn_myFlaskVirtualEnv/myflask/myapp/app/static/"
myapp.config.from_object(Config)

from app import routes
