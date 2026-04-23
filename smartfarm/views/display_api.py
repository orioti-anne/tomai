from flask import Blueprint, request, jsonify
from smartfarm import db
from smartfarm.models import PredictionDisplay

bp = Blueprint('display_api', __name__)

@bp.route('/api/receive-prediction', methods=['POST'])
def receive_prediction():
    data = request.get_json()

    if not data:
        return jsonify({"status": "fail", "message": "No data"}), 400

    try:
        new_entry = PredictionDisplay(
            category=data.get('type'),
            result_value=data.get('value'),
            target_date=data.get('target_date'),
            raw_json=str(data)
        )
        db.session.add(new_entry)
        db.session.commit()

        print(f"[GCP 수신완료] {data.get('type')}: {data.get('value')}")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "fail", "error": str(e)}), 500