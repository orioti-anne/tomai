import os
from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

import config

db = SQLAlchemy()
migrate = Migrate()


def create_app(enable_scheduler: bool = True):
    app = Flask(__name__)
    app.config.from_object(config)

    db.init_app(app)
    migrate.init_app(app, db)

    from . import models

    from .views import (
        main_views,
        question_views,
        answer_views,
        auth_views,
        farm_views,
        prediction_views,
        monitoring_views,
        env_api_views,
        growth_views,
        env_control_views,
        dashboard_views,
        display_api
    )

    app.register_blueprint(main_views.bp)
    app.register_blueprint(question_views.bp)
    app.register_blueprint(answer_views.bp)
    app.register_blueprint(auth_views.bp)
    app.register_blueprint(farm_views.bp)
    app.register_blueprint(prediction_views.bp)
    app.register_blueprint(monitoring_views.bp)
    app.register_blueprint(env_api_views.bp)
    app.register_blueprint(growth_views.bp)
    app.register_blueprint(env_control_views.bp)
    app.register_blueprint(dashboard_views.bp)
    app.register_blueprint(display_api.bp)

    if enable_scheduler:
        if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
            try:
                from smartfarm.services.env_scheduler import init_scheduler
                init_scheduler(app)
                print("[SCHEDULER] 통합 스케줄러(환경/가격/기상) 가동 완료")
            except Exception as e:
                print(f"[SCHEDULER] 실행 실패: {e}")
        else:
            print("[SCHEDULER] reloader 부모 프로세스에서는 스케줄러를 실행하지 않습니다.")
    else:
        print("[SCHEDULER] enable_scheduler=False 상태입니다.")

    return app