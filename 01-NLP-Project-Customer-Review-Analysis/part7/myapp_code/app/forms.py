from flask_wtf import FlaskForm
from wtforms import SelectField, StringField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class mySelectForm(FlaskForm):
    field1 = SelectField()
    submit = SubmitField('Submit')
