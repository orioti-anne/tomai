from flask import Blueprint, url_for, render_template, request, flash, session, g, redirect
from smartfarm import db
from smartfarm.forms import FarmCreationForm, CultivationForm
from smartfarm.models import Farms, Cultivations
from datetime import datetime

bp = Blueprint('farm', __name__, url_prefix='/farm')

@bp.route('/register/', methods=('GET', 'POST'))
def register():
    form = FarmCreationForm()

    if request.method == 'POST':
        region_l1_val = request.form.get('region_l1')
        region_l2_val = request.form.get('region_l2')
        if region_l1_val:
            form.region_l1.choices = [(region_l1_val, region_l1_val)]
        if region_l2_val:
            form.region_l2.choices = [(region_l2_val, region_l2_val)]

    if not g.user:
        flash("로그인이 필요한 서비스입니다.")
        return redirect(url_for('auth.login'))

    if form.validate_on_submit():
        now = datetime.now()
        new_farm = Farms(
            user_id=g.user.user_id,
            farm_name=form.farm_name.data,
            region_l1=form.region_l1.data,
            region_l2=form.region_l2.data,
            total_area=form.total_area.data,
            first_survey_year=now.year,
            created_at=now
        )

        try:
            db.session.add(new_farm)
            db.session.commit()

            action = request.form.get('action')
            if action == 'save_and_more':
                flash(f"'{form.farm_name.data}' 등록 완료! 다음 농가를 입력해주세요.")
                return redirect(url_for('farm.register'))
            else:
                flash(f"'{form.farm_name.data}' 등록이 완료되었습니다.")
                return redirect(url_for('farm.list'))

        except Exception as e:
            db.session.rollback()
            flash(f"오류 발생: {str(e)}")

    return render_template('farm/farm_form.html', form=form)


@bp.route('/list/')
def list():
    if not g.user:
        return redirect(url_for('auth.login'))

    farm_list = Farms.query.filter_by(user_id=g.user.user_id).order_by(Farms.created_at.desc()).all()
    return render_template('farm/farm_list.html', farm_list=farm_list)


@bp.route('/toggle_status/<int:farm_id>')
def toggle_status(farm_id):
    farm = Farms.query.get_or_404(farm_id)
    if g.user.user_id != farm.user_id:
        flash("권한이 없습니다.")
        return redirect(url_for('farm.list'))
    farm.is_active = 'N' if farm.is_active == 'Y' else 'Y'

    try:
        db.session.commit()
        status_msg = "비활성화" if farm.is_active == 'N' else "활성화"
        flash(f"'{farm.farm_name}' 농가가 {status_msg} 되었습니다.")
    except Exception as e:
        db.session.rollback()
        flash(f"오류 발생: {str(e)}")

    return redirect(url_for('farm.list'))


@bp.route('/cultivation/register/<int:farm_id>', methods=('GET', 'POST'))
def register_cult(farm_id):
    if not g.user:
        return redirect(url_for('auth.login'))

    farm = Farms.query.get_or_404(farm_id)
    form = CultivationForm()

    if request.method == 'POST' and form.validate_on_submit():
        planting_date = form.planting_date.data
        survey_year = planting_date.year
        now = datetime.now()
        cult_name = f"{farm.farm_name} {survey_year}년 {form.item.data} {form.crop_cycle.data}작기"

        try:
            input_planting_area = float(form.planting_area.data or 0)
        except Exception:
            flash('식부면적 형식이 올바르지 않습니다.')
            return render_template('farm/cult_form.html', form=form, farm=farm)

        if input_planting_area <= 0:
            flash('식부면적은 0보다 커야 합니다.')
            return render_template('farm/cult_form.html', form=form, farm=farm)

        farm_total_area = float(farm.total_area or 0)

        if farm_total_area > 0 and input_planting_area > farm_total_area:
            flash(
                f"식부면적({input_planting_area}㎡)은 농가 총면적({farm_total_area}㎡)을 초과할 수 없습니다."
            )
            return render_template('farm/cult_form.html', form=form, farm=farm)

        new_cult = Cultivations(
            farm_id=farm_id,
            cult_name=cult_name,
            item=form.item.data,
            item_variety=form.item_variety.data,
            crop_cycle=int(form.crop_cycle.data),
            planting_date=form.planting_date.data,
            planting_area=input_planting_area,
            planting_density=form.planting_density.data,
            house_type=form.house_type.data,
            house_form=form.house_form.data,
            status='active',
            survey_year=survey_year,
            created_at=now
        )

        try:
            db.session.add(new_cult)
            db.session.commit()
            flash(f"'{cult_name}' 재배 정보 등록이 완료되었습니다.")
            return redirect(url_for('farm.list'))
        except Exception as e:
            db.session.rollback()
            flash(f"등록 중 오류가 발생했습니다: {str(e)}")

    return render_template('farm/cult_form.html', form=form, farm=farm)


@bp.route('/toggle_cult_status/<int:cult_id>')
def toggle_cult_status(cult_id):
    cult = Cultivations.query.get_or_404(cult_id)
    if cult.status in ['closed', 'hidden']:
        cult.status = 'active'
    else:
        cult.status = 'closed'
    db.session.commit()
    return redirect(url_for('farm.list'))


@bp.route('/hide_cult/<int:cult_id>')
def hide_cult(cult_id):
    cult = Cultivations.query.get_or_404(cult_id)
    cult.status = 'hidden'
    db.session.commit()
    return redirect(url_for('farm.list'))
