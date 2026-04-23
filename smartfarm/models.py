from smartfarm import db
from datetime import datetime


class Users(db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, db.Sequence('users_user_id_seq'), primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=True)
    address = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)


class Question(db.Model):
    __tablename__ = 'question'
    id = db.Column(db.Integer, db.Sequence('question_id_seq'), primary_key=True)
    subject = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text(), nullable=False)
    created_date = db.Column(db.DateTime(), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    user = db.relationship('Users', backref=db.backref('question_set', lazy=True))



class Answer(db.Model):
    __tablename__ = 'answer'
    id = db.Column(db.Integer, db.Sequence('answer_id_seq'), primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('question.id', ondelete='CASCADE'), nullable=False)
    question = db.relationship('Question',backref=db.backref('answers_set', cascade='all, delete-orphan', lazy=True))
    content = db.Column(db.Text(), nullable=False)
    created_date = db.Column(db.DateTime(), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    user = db.relationship('Users', backref=db.backref('answer_set', lazy=True))

def get_current_year():
    return datetime.now().year


class Farms(db.Model):
    __tablename__ = 'farms'
    farm_id = db.Column(db.Integer, db.Sequence('farms_farm_id_seq'), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    farm_name = db.Column(db.String(100), nullable=False)
    farm_num = db.Column(db.Integer, nullable=True)
    region_l1 = db.Column(db.String(50))
    region_l2 = db.Column(db.String(50))
    total_area = db.Column(db.Float)
    first_survey_year = db.Column(db.Integer, default=get_current_year)
    created_at = db.Column(db.DateTime, default=datetime.now)
    is_active = db.Column(db.String(1), default='Y')
    user = db.relationship('Users', backref=db.backref('farm_set'))
    cultivations = db.relationship('Cultivations', backref='farm', cascade='all, delete-orphan', lazy=True)


class Cultivations(db.Model):
    __tablename__ = 'cultivations'
    cult_id = db.Column(db.Integer, db.Sequence('cultivations_cult_id_seq'), primary_key=True)
    farm_id = db.Column(db.Integer, db.ForeignKey('farms.farm_id'), nullable=False)
    item = db.Column(db.String(100))
    item_variety = db.Column(db.String(100))
    crop_cycle = db.Column(db.Integer)
    planting_date = db.Column(db.Date)
    planting_area = db.Column(db.Float)
    planting_density = db.Column(db.Float)
    house_type = db.Column(db.String(50))
    house_form = db.Column(db.String(50))
    status = db.Column(db.String(20), default='active')
    survey_year = db.Column(db.Integer, default=get_current_year)
    created_at = db.Column(db.DateTime, default=datetime.now)
    cult_name = db.Column(db.String(200))
    virtual_sensor_enabled = db.Column(db.String(1), nullable=False, default='N')

    def __repr__(self):
        return f'<Cultivations ID:{self.cult_id} | {self.farm.farm_name} | {self.item} - {self.survey_year}>'


class Products(db.Model):
    __tablename__ = 'products'
    product_id = db.Column(db.Integer, db.Sequence('products_product_id_seq'), primary_key=True)
    cult_id = db.Column(db.Integer, db.ForeignKey('cultivations.cult_id'), nullable=False)
    production_date = db.Column(db.Date)
    total_quantity = db.Column(db.Float)
    total_sales = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.now)
    cultivation = db.relationship('Cultivations', backref=db.backref('products_set', lazy=True))


class ProdSummary(db.Model):
    __tablename__ = 'prod_summary'
    prodsu_id = db.Column(db.Integer, db.Sequence('prod_summary_prodsu_id_seq'), primary_key=True)
    cult_id = db.Column(db.Integer, db.ForeignKey('cultivations.cult_id'), nullable=False)
    cult_end_date = db.Column(db.Date)
    cult_total_quantity = db.Column(db.Float)
    cult_total_sales = db.Column(db.Float)
    yield_per_area = db.Column(db.Float)
    unit_price = db.Column(db.Float)
    origin_type = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.now)
    cultivation = db.relationship('Cultivations', backref=db.backref('prod_summary_set', lazy=True))

    def __repr__(self):
        return f'<ProdSummary ID:{self.prodsu_id} | 'f'CultID:{self.cult_id} | 'f'EndDate:{self.cult_end_date}>'


class Environment(db.Model):
    __tablename__ = 'environment'
    env_id = db.Column(db.Integer, db.Sequence('seq_env_id'), primary_key=True)
    cult_id = db.Column(db.Integer, db.ForeignKey('cultivations.cult_id'), nullable=False)
    measure_time = db.Column(db.DateTime, nullable=False)
    out_temp = db.Column(db.Float)
    out_wind_direction = db.Column(db.Float)
    out_wind_speed = db.Column(db.Float)
    out_solar_rad = db.Column(db.Float)
    out_acc_solar_rad = db.Column(db.Float)
    rain_detection = db.Column(db.Integer)
    in_temp = db.Column(db.Float)
    in_humidity = db.Column(db.Float)
    in_co2 = db.Column(db.Float)
    soil_temp = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.now)
    cultivation = db.relationship('Cultivations', backref=db.backref('env_set', lazy=True))

    def __repr__(self):
        return f'<Environment {self.measure_time} - CultID: {self.cult_id}>'


class EnvCleaned(db.Model):
    __tablename__ = 'env_cleaned'
    envcl_id = db.Column(db.Integer, db.Sequence('env_cleaned_envcl_id_seq'), primary_key=True)
    env_id = db.Column(db.Integer, db.ForeignKey('environment.env_id'), nullable=False)
    cult_id = db.Column(db.Integer, db.ForeignKey('cultivations.cult_id'), nullable=False)
    measure_time = db.Column(db.DateTime, nullable=False)
    out_temp = db.Column(db.Float)
    out_wind_direction = db.Column(db.Float)
    out_wind_speed = db.Column(db.Float)
    out_solar_rad = db.Column(db.Float)
    out_acc_solar_rad = db.Column(db.Float)
    rain_detection = db.Column(db.Integer)
    in_temp = db.Column(db.Float)
    in_humidity = db.Column(db.Float)
    in_co2 = db.Column(db.Float)
    soil_temp = db.Column(db.Float)
    out_acc_solar_rad_status = db.Column(db.Integer, default=0)
    in_temp_status = db.Column(db.Integer, default=0)
    in_humidity_status = db.Column(db.Integer, default=0)
    in_co2_status = db.Column(db.Integer, default=0)
    measure_date = db.Column(db.Date)
    measure_hour = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.now)


class EnvSummary(db.Model):
    __tablename__ = 'env_summary'
    envsu_id = db.Column(db.Integer, db.Sequence('env_summary_envsu_id_seq'), primary_key=True)
    cult_id = db.Column(db.Integer, db.ForeignKey('cultivations.cult_id'), nullable=False)
    measure_date = db.Column(db.Date, nullable=False)
    daily_out_temp = db.Column(db.Float)
    daily_acc_solar = db.Column(db.Float)
    daily_rain_detection = db.Column(db.Integer)
    daily_in_temp = db.Column(db.Float)
    daily_in_humidity = db.Column(db.Float)
    daily_in_co2 = db.Column(db.Float)
    daily_soil_temp = db.Column(db.Float)
    acc_temp = db.Column(db.Float)
    acc_solar = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.now)
    cultivation = db.relationship('Cultivations', backref=db.backref('env_summary_set', lazy=True))


class Growth(db.Model):
    __tablename__ = 'growth'
    growth_id = db.Column(db.Integer, db.Sequence('growth_growth_id_seq'), primary_key=True)
    cult_id = db.Column(db.Integer, db.ForeignKey('cultivations.cult_id'), nullable=False)
    inspect_date = db.Column(db.Date, nullable=False)
    plant_num = db.Column(db.Integer)
    branch_num = db.Column(db.Integer)
    plant_height = db.Column(db.Float)
    growth_length = db.Column(db.Float)
    leaf_count = db.Column(db.Integer)
    leaf_length = db.Column(db.Float)
    leaf_width = db.Column(db.Float)
    branch_width = db.Column(db.Float)
    cluster_height = db.Column(db.Float)
    cluster_num = db.Column(db.Integer)
    flowers_per_cluster = db.Column(db.Float)
    blooming_per_cluster = db.Column(db.Float)
    fruits_per_cluster = db.Column(db.Float)
    blooming_group = db.Column(db.Float)
    fruiting_group = db.Column(db.Float)
    remarks = db.Column(db.String(1000))
    created_at = db.Column(db.DateTime, default=datetime.now)
    cultivation = db.relationship('Cultivations', backref=db.backref('growth_set', lazy=True))

    def __repr__(self):
        return f'<Growth ID:{self.growth_id} | CultID:{self.cult_id} | Date:{self.inspect_date}>'


class GrowSummary(db.Model):
    __tablename__ = 'grow_summary'
    growsu_id = db.Column(db.Integer, db.Sequence('grow_summary_growsu_id_seq'), primary_key=True)
    growth_id = db.Column(db.Integer, db.ForeignKey('growth.growth_id'), nullable=True)
    cult_id = db.Column(db.Integer, db.ForeignKey('cultivations.cult_id'), nullable=False)
    inspect_date = db.Column(db.Date, nullable=False)
    growth_days = db.Column(db.Integer)
    plant_num = db.Column(db.Integer)
    branch_num = db.Column(db.Integer)
    plant_height = db.Column(db.Float)
    growth_length = db.Column(db.Float)
    leaf_count = db.Column(db.Integer)
    leaf_length = db.Column(db.Float)
    leaf_width = db.Column(db.Float)
    branch_width = db.Column(db.Float)
    cluster_height = db.Column(db.Float)
    cluster_num = db.Column(db.Integer)
    flowers_per_cluster = db.Column(db.Float)
    blooming_per_cluster = db.Column(db.Float)
    fruits_per_cluster = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.now)
    origin_type = db.Column(db.Integer, default=0)
    growth = db.relationship('Growth', backref=db.backref('summary_set', lazy=True))
    cultivation = db.relationship('Cultivations', backref=db.backref('grow_summary_set', lazy=True))

    def __repr__(self):
        return f'<GrowSummary ID:{self.growsu_id} | CultID:{self.cult_id} | Date:{self.inspect_date}>'


class PredictionResults(db.Model):
    __tablename__ = 'prediction_results'
    prediction_id = db.Column(db.Integer, db.Sequence('prediction_results_prediction_id_seq'), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    farm_id = db.Column(db.Integer, db.ForeignKey('farms.farm_id'), nullable=False)
    cult_id = db.Column(db.Integer, db.ForeignKey('cultivations.cult_id'), nullable=False)
    prediction_date = db.Column(db.DateTime, default=datetime.now, nullable=False)
    item = db.Column(db.String(100))
    item_variety = db.Column(db.String(100))
    crop_cycle = db.Column(db.Integer)
    planting_date = db.Column(db.Date)
    expected_harvest_date = db.Column(db.Date)
    planting_area = db.Column(db.Float)
    planting_density = db.Column(db.Float)
    house_type = db.Column(db.String(50))
    house_form = db.Column(db.String(50))
    avg_days_to_peak_harvest = db.Column(db.Integer)
    avg_yield_per_m2 = db.Column(db.Float)
    expected_quantity = db.Column(db.Float)
    expected_price_per_kg = db.Column(db.Float)
    expected_sales = db.Column(db.Float)
    sample_count = db.Column(db.Integer)
    latest_market_price = db.Column(db.Float)
    market_name = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    price_day_95 = db.Column(db.Float)
    price_day_105 = db.Column(db.Float)
    price_day_115 = db.Column(db.Float)
    prediction_source = db.Column(db.String(20))
    prediction_confidence = db.Column(db.String(20))
    prediction_message = db.Column(db.Text)
    user = db.relationship('Users', backref=db.backref('prediction_results_set', lazy=True))
    farm = db.relationship('Farms', backref=db.backref('prediction_results_set', lazy=True))
    cultivation = db.relationship('Cultivations', backref=db.backref('prediction_results_set', lazy=True))

    def __repr__(self):
        return f'<PredictionResults ID:{self.prediction_id} | 'f'CultID:{self.cult_id} | 'f'Harvest:{self.expected_harvest_date}>'


class PredictionDisplay(db.Model):
    __tablename__ = 'prediction_display'
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(50))   # 'price', 'weather', 'growth'
    result_value = db.Column(db.Float)    # 예측 또는 수집된 수치
    target_date = db.Column(db.String(20)) # 데이터 기준 날짜
    created_at = db.Column(db.DateTime, default=datetime.now)
    raw_json = db.Column(db.Text)          # 전체 데이터 백업용

    def __repr__(self):
        return f'<PredictionDisplay {self.category}: {self.result_value}>'