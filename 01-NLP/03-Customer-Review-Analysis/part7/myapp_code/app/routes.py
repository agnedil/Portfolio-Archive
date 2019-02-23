# this explains how Python can access the static folder (look for url = os.path.join(current_app.root_path, current_app.static_folder, filename))
# https://www.reddit.com/r/flask/comments/2i102e/af_how_to_open_static_file_specifically_json/
from __future__ import division
import numpy as np
import os
import io
import getdata
from flask import render_template, flash, redirect, url_for, send_file, Response
from app import myapp
from app.forms import mySelectForm
from collections import defaultdict
from operator import itemgetter


# choices for SelectFields in forms
CUISINES = ('Chinese', 'Mediterranean', 'Italian', 'Mexican', 'Indian')
DISHES_1 = ('wonton soup', 'peking duck', 'mongolian beef', 'kung pao chicken', 'rice cake')
DISHES_2 = ('lobster bisque', 'greek salad', 'chicken kabob', 'basmati rice', 'baba ghanoush')
DISHES_3 = ('caprese salad', 'lobster ravioli', 'eggplant parm', 'margherita pizza', 'tiramisu')
DISHES_4 = ('tortilla soup', 'carne asada', 'chicken fajita', 'burrito', 'shrimp enchilada')
DISHES_5 = ('chicken tikka masala', 'tandoori chicken', 'lamb vindaloo', 'chana masala', 'garlic naan')


CUISINE_CHOICES = [(item, item) for item in CUISINES]
DISH_CHOICES_1 = [(item, item.capitalize()) for item in DISHES_1]
DISH_CHOICES_2 = [(item, item.capitalize()) for item in DISHES_2]
DISH_CHOICES_3 = [(item, item.capitalize()) for item in DISHES_3]
DISH_CHOICES_4 = [(item, item.capitalize()) for item in DISHES_4]
DISH_CHOICES_5 = [(item, item.capitalize()) for item in DISHES_5]

# these variables store user's selections
selectedCuisine = ''
selectedDish = ''
DISH_CHOICES = ''

@myapp.route('/')

# index (starting) page
@myapp.route('/index')
def index():
    user = {'username': 'Andrew'}
    return render_template('index.html', title='Home', user=user)

# page to select cuisine
@myapp.route('/selectc', methods=['GET', 'POST'])
def selectc():
    global selectedCuisine, DISH_CHOICES
    form = mySelectForm()    
    form.field1.label = 'Select cuisine:'
    form.field1.choices = CUISINE_CHOICES    
    if form.validate_on_submit():
        selectedCuisine = form.field1.data
        if selectedCuisine == 'Chinese': DISH_CHOICES = DISH_CHOICES_1
        if selectedCuisine == 'Mediterranean': DISH_CHOICES = DISH_CHOICES_2
        if selectedCuisine == 'Italian': DISH_CHOICES = DISH_CHOICES_3
        if selectedCuisine == 'Mexican': DISH_CHOICES = DISH_CHOICES_4
        if selectedCuisine == 'Indian': DISH_CHOICES = DISH_CHOICES_5
        #flash('You requested recommendations for {}'.format(selectedCuisine))        
        return redirect(url_for('selectd'))
    return render_template('selectc.html',  title='Please make your selections', form=form)

# page to select dishes if SelectedCuisine == 'italian'
@myapp.route('/selectd', methods=['GET', 'POST'])
def selectd():
    global selectedDish, DISH_CHOICES
    form = mySelectForm()    
    form.field1.label = 'Select dish:'    
    form.field1.choices = DISH_CHOICES
    if form.validate_on_submit():
        selectedDish = form.field1.data
        rawDir = os.path.join(myapp.root_path, myapp.static_folder)
        getdata.getData(selectedCuisine, selectedDish, rawDir)        
        return redirect(url_for('result'))    
    return render_template('selectd.html',  title='Please make your selections', form=form)

# page to show the results
@myapp.route('/result')
def result():
    message = 'Recommendations for dish {!r}, cuisine {!r}'.format(str(selectedDish), str(selectedCuisine))            
    return render_template('result.html', title='Results', text=selectedCuisine, message=message)

# THIS VIEW IS TO MAKE SURE THE BROWSER DOES NOT CACHE THE SAME DATA.TSV FILE ALL THE TIME!!!!
response = Response()
@myapp.after_request
def after_request(response):        
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
