from flask import Blueprint, url_for, render_template, request, flash, session, g
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import redirect

from smartfarm import db
from smartfarm.forms import UserCreationForm, UserLoginForm
from smartfarm.models import Users, Farms

bp = Blueprint('auth', __name__, url_prefix='/auth')


@bp.route('/signup/', methods=('GET', 'POST'))
def signup():
    form = UserCreationForm()
    if request.method == 'POST' and form.validate_on_submit():
        user = Users.query.filter_by(username=form.username.data).first()
        phone_check = Users.query.filter_by(phone=form.phone.data).first()

        if not user and not phone_check:
            new_user = Users(
                username=form.username.data,
                password=generate_password_hash(form.password1.data),
                name=form.name.data,
                phone=form.phone.data,
                email=form.email.data,
                address=form.address.data
            )
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('main.index'))
        else:
            if user:
                flash('이미 존재하는 아이디입니다.')
            if phone_check:
                flash('이미 등록된 휴대폰 번호입니다.')

    return render_template('auth/signup.html', form=form)


@bp.route('/login/', methods=('GET', 'POST'))
def login():
    form = UserLoginForm()
    if request.method == 'POST' and form.validate_on_submit():
        error = None
        user = Users.query.filter_by(username=form.username.data).first()
        if not user:
            error = "존재하지 않는 사용자입니다"
        elif not check_password_hash(user.password, form.password.data):
            error = "비밀번호가 올바르지 않습니다"

        if error is None:
            session.clear()
            session['user_id'] = user.user_id
            return redirect(url_for('main.index'))
        flash(error)
    return render_template('auth/login.html', form=form)


@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = Users.query.get(user_id)


@bp.route('/logout/')
def logout():
    session.clear()
    return redirect(url_for('main.index'))


@bp.route('/withdraw/', methods=['POST'])
def withdraw():
    if g.user:
        Farms.query.filter_by(user_id=g.user.user_id).update({'user_id': 0})
        user_farms = Farms.query.filter_by(user_id=0).all()
        for farm in user_farms:
            if farm.farm_name == f"{g.user.name}님의 농장":
                farm.farm_name = f"Farm_{farm.farm_id}"
        db.session.delete(g.user)
        db.session.commit()
        session.clear()

        flash("탈퇴가 완료되었습니다. 데이터는 비식별화되어 보관됩니다.")
    return redirect(url_for('main.index'))