from smartfarm import db
from datetime import datetime


class Users(db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, db.Sequence('users_seq', start=1, increment=1), primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=True)
    address = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)

class Question(db.Model):
    __tablename__ = 'question'
    id = db.Column(db.Integer, db.Sequence('question_seq', start=1, increment=1), primary_key=True)
    subject = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text(), nullable=False)
    created_date = db.Column(db.DateTime(), nullable=False)

class Answer(db.Model):
    __tablename__ = 'answer'
    id = db.Column(db.Integer, db.Sequence('answer_seq', start=1, increment=1), primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('question.id', ondelete='CASCADE'))
    question = db.relationship('Question', backref=db.backref('answers_set', cascade='all, delete-orphan'))
    content = db.Column(db.Text(), nullable=False)
    created_date = db.Column(db.DateTime(), nullable=False)

def get_current_year():
    return datetime.now().year

class Farms(db.Model):
    __tablename__ = 'farms'
    farm_id = db.Column(db.Integer, db.Sequence('farms_seq'), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    farm_name = db.Column(db.String(100), nullable=False)
    farm_num = db.Column(db.Integer, nullable=True)
    region_l1 = db.Column(db.String(50))
    region_l2 = db.Column(db.String(50))
    total_area = db.Column(db.Float)
    first_survey_year = db.Column(db.Integer, default=get_current_year)
    created_at = db.Column(db.DateTime, default=datetime.now)
    user = db.relationship('Users', backref=db.backref('farm_set'))


class Cultivations(db.Model):
    __tablename__ = 'cultivations'
    cult_id = db.Column(db.Integer, db.Sequence('cultivations_seq'), primary_key=True)
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
    farm = db.relationship('Farms', backref=db.backref('cultivations', lazy=True))

    def __repr__(self):
        return f'<Cultivations ID:{self.cult_id} | {self.farm.farm_name} | {self.item} - {self.survey_year}>'


class Products(db.Model):
    __tablename__ = 'products'
    product_id = db.Column(db.Integer, db.Sequence('products_seq'), primary_key=True)
    cult_id = db.Column(db.Integer, db.ForeignKey('cultivations.cult_id'), nullable=False)
    production_date = db.Column(db.Date)
    total_quantity = db.Column(db.Float)
    total_sales = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.now)
    cultivation = db.relationship('Cultivations', backref=db.backref('products_set', lazy=True))


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


class Growth(db.Model):
    __tablename__ = 'growth'
    growth_id = db.Column(db.Integer, db.Sequence('growth_seq'), primary_key=True)
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