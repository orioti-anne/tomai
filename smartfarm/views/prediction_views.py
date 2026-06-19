import os
import json
from datetime import datetime

from flask import Blueprint, request, jsonify, g, redirect, url_for, render_template, flash

from smartfarm import db
from smartfarm.models import Farms, Cultivations, PredictionResults
from smartfarm.services.prediction_service import run_default_prediction, run_ml_prediction, run_environment_recommendation

bp = Blueprint('prediction', __name__, url_prefix='/prediction')


@bp.route('/environment/recommend', methods=['POST'])
def recommend_environment_api():
    if not g.user:
        return jsonify({
            'success': False,
            'message': '로그인이 필요합니다.'
        }), 401

    data = request.get_json(silent=True) or {}

    try:
        cult_id = data.get("cult_id")

        sensor_data = {
            "temp": data.get("temp"),
            "humid": data.get("humid"),
            "co2": data.get("co2"),
            "solar": data.get("solar"),
            "soil_temp": data.get("soil_temp"),
            "dap": data.get("dap"),
            "high_temp_hours": data.get("high_temp_hours", 0),
        }

        if cult_id and not sensor_data.get("dap"):
            cult = Cultivations.query.filter_by(cult_id=int(cult_id)).first()
            if cult and cult.planting_date:
                sensor_data["dap"] = (datetime.today().date() - cult.planting_date).days

        result = run_environment_recommendation(sensor_data)

        return jsonify({
            "success": True,
            "result": result,
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e),
        }), 400


@bp.route("/run/<int:cult_id>")
def run_ml_prediction_view(cult_id):
    return redirect(url_for('prediction.prediction'))


@bp.route('/run', methods=['POST'])
def run_prediction_api():
    if not g.user:
        return jsonify({
            'success': False,
            'message': '로그인이 필요합니다.'
        }), 401

    data = request.get_json(silent=True) or {}

    try:
        farm_id = int(data.get('farm_id'))
        cult_id = data.get('cult_id')

        req_planting_date = data.get('planting_date')
        req_item = data.get('item')
        req_crop_cycle = data.get('crop_cycle')
        req_item_variety = data.get('item_variety')
        req_planting_area = data.get('planting_area')
        req_planting_density = data.get('planting_density')
        req_house_type = data.get('house_type')
        req_house_form = data.get('house_form')

        cult_name = None

        planting_date = None
        item = None
        crop_cycle = None
        item_variety = None
        planting_area = None
        planting_density = None
        house_type = None
        house_form = None

        farm = Farms.query.filter_by(
            farm_id=farm_id,
            user_id=g.user.user_id
        ).first()

        if not farm:
            raise ValueError('농가 정보를 찾을 수 없습니다.')

        # 기존 재배정보 선택
        if cult_id:
            cult_id = int(cult_id)

            cult = Cultivations.query.filter_by(
                cult_id=cult_id,
                farm_id=farm_id
            ).first()

            if not cult:
                raise ValueError('재배 정보를 찾을 수 없습니다.')

            if cult.status == 'hidden':
                raise ValueError('사용할 수 없는 재배 정보입니다.')

            item = cult.item
            crop_cycle = cult.crop_cycle
            item_variety = cult.item_variety
            planting_date = cult.planting_date.strftime('%Y-%m-%d') if cult.planting_date else None
            planting_area = float(cult.planting_area) if cult.planting_area is not None else None
            planting_density = float(cult.planting_density) if cult.planting_density is not None else None
            house_type = cult.house_type
            house_form = cult.house_form
            cult_name = cult.cult_name

        # 새 재배정보 직접 입력
        else:
            planting_date = req_planting_date
            item = req_item
            crop_cycle = req_crop_cycle
            item_variety = req_item_variety
            planting_area = req_planting_area
            planting_density = req_planting_density
            house_type = req_house_type
            house_form = req_house_form

            try:
                input_planting_area = float(planting_area or 0)
            except Exception:
                raise ValueError('식부면적 형식이 올바르지 않습니다.')

            if input_planting_area <= 0:
                raise ValueError('식부면적은 0보다 커야 합니다.')

            farm_total_area = float(farm.total_area or 0)
            if farm_total_area > 0 and input_planting_area > farm_total_area:
                raise ValueError(
                    f'식부면적({input_planting_area}㎡)은 농가 총면적({farm_total_area}㎡)을 초과할 수 없습니다.'
                )

            cult_name_parts = []

            if farm.farm_name:
                cult_name_parts.append(farm.farm_name)

            if planting_date:
                try:
                    year = datetime.strptime(planting_date, '%Y-%m-%d').year
                    cult_name_parts.append(f'{year}년')
                except Exception:
                    pass

            if item:
                cult_name_parts.append(item)

            if crop_cycle:
                cult_name_parts.append(f'{crop_cycle}작기')

            cult_name = ' '.join(cult_name_parts)

            new_cult = Cultivations(
                farm_id=farm_id,
                item=item,
                item_variety=item_variety,
                crop_cycle=int(crop_cycle) if crop_cycle else None,
                planting_date=datetime.strptime(planting_date, '%Y-%m-%d').date()
                if planting_date else None,
                planting_area=input_planting_area,
                planting_density=float(planting_density) if planting_density else None,
                house_type=house_type,
                house_form=house_form,
                status='active',
                cult_name=cult_name
            )

            db.session.add(new_cult)
            db.session.commit()

            cult_id = new_cult.cult_id
            planting_area = input_planting_area

        import requests as _req
        MAC_API_URL = os.getenv("MAC_API_URL", "http://localhost:5001")
        MAC_API_KEY = os.environ["MAC_API_KEY"]
        try:
            mac_res = _req.post(
                f"{MAC_API_URL}/api/prediction/run",
                headers={"X-API-Key": MAC_API_KEY},
                json={
                    "cult_id": cult_id,
                    "user_id": g.user.user_id,
                    "farm_id": farm_id,
                    "planting_date": planting_date,
                    "item": item,
                    "crop_cycle": crop_cycle,
                    "item_variety": item_variety,
                    "planting_area": planting_area,
                    "planting_density": planting_density,
                    "house_type": house_type,
                    "house_form": house_form
                },
                timeout=15
            )
            if mac_res.status_code == 200:
                result = mac_res.json().get("result", {})
            else:
                raise Exception(f"API 오류: {mac_res.status_code}")
        except Exception as mac_e:
            print(f"[PREDICTION] API 실패, 로컬 fallback: {mac_e}")
            result = run_default_prediction(
                cult_id=cult_id,
                farm_id=farm_id,
                planting_date=planting_date,
                item=item,
                crop_cycle=crop_cycle,
                item_variety=item_variety,
                planting_area=planting_area,
                planting_density=planting_density,
                house_type=house_type,
                house_form=house_form
            )

        expected_harvest_date = None
        if result.get("expected_harvest_date"):
            expected_harvest_date = datetime.strptime(
                result["expected_harvest_date"], "%Y-%m-%d"
            ).date()

        prediction = PredictionResults(
            user_id=g.user.user_id,
            farm_id=farm_id,
            cult_id=cult_id,
            item=item,
            item_variety=item_variety,
            crop_cycle=int(crop_cycle) if crop_cycle else None,
            planting_date=datetime.strptime(planting_date, '%Y-%m-%d').date()
            if planting_date else None,
            expected_harvest_date=expected_harvest_date,
            planting_area=float(planting_area) if planting_area else None,
            planting_density=float(planting_density) if planting_density else None,
            house_type=house_type,
            house_form=house_form,
            avg_days_to_peak_harvest=result.get('avg_days_to_peak_harvest'),
            avg_yield_per_m2=result.get('avg_yield_per_m2'),
            expected_quantity=result.get('expected_quantity'),
            expected_price_per_kg=result.get('expected_price_per_kg'),
            expected_sales=result.get('expected_sales'),
            sample_count=result.get('sample_count'),
            latest_market_price=result.get('latest_market_price'),
            market_name='가락도매',
            price_day_95=result.get('price_day_95'),
            price_day_105=result.get('price_day_105'),
            price_day_115=result.get('price_day_115'),
            comparison_json=json.dumps(result.get('comparison_data', []), ensure_ascii=False),
            prediction_source = result.get('prediction_source'),
            prediction_confidence = result.get('prediction_confidence'),
            prediction_message = result.get('prediction_message'),
        )

        db.session.add(prediction)
        db.session.commit()

        return jsonify({
            'success': True,
            'results': result,
            'cult_id': cult_id,
            'cult_name': cult_name,
            'prediction_id': prediction.prediction_id
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400


@bp.route('/')
def prediction():
    if not g.user:
        return redirect(url_for('auth.login'))

    farms = (
        Farms.query
        .filter_by(user_id=g.user.user_id, is_active='Y')
        .all()
    )

    farm_data = []

    for farm in farms:
        cultivations = []

        for cult in farm.cultivations:
            if cult.status == 'hidden':
                continue

            cultivations.append({
                'cult_id': cult.cult_id,
                'cult_name': cult.cult_name or '',
                'item': cult.item or '',
                'crop_cycle': int(cult.crop_cycle) if cult.crop_cycle is not None else '',
                'item_variety': cult.item_variety or '',
                'planting_date': cult.planting_date.strftime('%Y-%m-%d') if cult.planting_date else '',
                'planting_area': float(cult.planting_area) if cult.planting_area is not None else '',
                'planting_density': float(cult.planting_density) if cult.planting_density is not None else '',
                'house_type': cult.house_type or '',
                'house_form': cult.house_form or '',
                'status': cult.status or 'active'
            })

        farm_data.append({
            'farm_id': farm.farm_id,
            'farm_name': farm.farm_name or '',
            'cultivations': cultivations
        })

    return render_template(
        'prediction.html',
        farm_list=farms,
        farm_data_json=json.dumps(farm_data, ensure_ascii=False)
    )