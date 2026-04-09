from flask import Blueprint, request, jsonify

from .. import db
from ..models import Cultivations

bp = Blueprint(
    "environment_control",
    __name__,
    url_prefix="/api/environment-control"
)


@bp.route("/virtual-sensor/toggle", methods=["POST"])
def toggle_virtual_sensor():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "요청 데이터가 없습니다."
            }), 400

        cult_id = data.get("cult_id")
        enabled = data.get("enabled")

        if cult_id is None or enabled is None:
            return jsonify({
                "success": False,
                "message": "cult_id 또는 enabled 값이 없습니다."
            }), 400

        cultivation = Cultivations.query.filter_by(cult_id=cult_id).first()

        if not cultivation:
            return jsonify({
                "success": False,
                "message": "해당 재배 정보를 찾을 수 없습니다."
            }), 404

        cultivation.virtual_sensor_enabled = "Y" if enabled else "N"
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "가상센서 상태가 변경되었습니다.",
            "cult_id": cult_id,
            "virtual_sensor_enabled": cultivation.virtual_sensor_enabled
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            "success": False,
            "message": f"서버 오류: {str(e)}"
        }), 500