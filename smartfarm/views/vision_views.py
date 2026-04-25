from flask import Blueprint, render_template, g, redirect, url_for
from smartfarm.models import Cultivations, Farms

bp = Blueprint('vision', __name__, url_prefix='/vision')

@bp.route('/')
def index():
    if not g.user:
        return redirect(url_for('auth.login'))

    cult_list = (
        Cultivations.query
        .join(Farms, Cultivations.farm_id == Farms.farm_id)
        .filter(
            Farms.user_id == g.user.user_id,
            Cultivations.status == 'active'
        )
        .order_by(Cultivations.created_at.desc())
        .all()
    )

    selected_cult = cult_list[0] if cult_list else None

    return render_template('vision.html',
        cult_list=cult_list,
        selected_cult=selected_cult,
        username=g.user.username
    )