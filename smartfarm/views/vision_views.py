from flask import Blueprint, render_template, g
from smartfarm.models import Cultivations

bp = Blueprint('vision', __name__, url_prefix='/vision')

@bp.route('/')
def index():
    if not g.user:
        from flask import redirect, url_for
        return redirect(url_for('auth.login'))

    cult_list = Cultivations.query.filter_by(
        user_id=g.user.user_id, status='active'
    ).order_by(Cultivations.created_at.desc()).all()

    selected_cult = cult_list[0] if cult_list else None

    return render_template('vision.html',
        cult_list=cult_list,
        selected_cult=selected_cult,
        username=g.user.username
    )