from flask import Blueprint, render_template, url_for, redirect, g, request
from datetime import date

from smartfarm.models import Cultivations, PredictionResults, Farms, Growth
from smartfarm.services.weather_service import get_weather

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return render_template('home.html')



@bp.route('/question')
def question_list():
    return redirect(url_for('question._list'))