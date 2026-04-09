from flask import Blueprint, request, jsonify

from smartfarm.services.env_api_service import (
    ingest_environment_one,
    ingest_environment_bulk,
)

bp = Blueprint("env_api", __name__, url_prefix="/api/environment")


@bp.route("/ingest", methods=["POST"])
def ingest_environment():
    data = request.get_json(silent=True) or {}

    try:
        result = ingest_environment_one(data)
        return jsonify({
            "success": True,
            "message": "환경 데이터가 저장되었습니다.",
            "result": result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 400


@bp.route("/ingest/bulk", methods=["POST"])
def ingest_environment_bulk_view():
    data = request.get_json(silent=True) or []

    if not isinstance(data, list):
        return jsonify({
            "success": False,
            "message": "배치 데이터는 배열(list) 형태여야 합니다."
        }), 400

    try:
        result = ingest_environment_bulk(data)
        return jsonify({
            "success": True,
            "message": "환경 배치 데이터가 저장되었습니다.",
            "result": result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 400