from flask_wtf import FlaskForm
from wtforms.fields.simple import StringField, TextAreaField, PasswordField
from wtforms.fields import EmailField, FloatField, SelectField
from wtforms.validators import DataRequired, Length, Email, EqualTo, NumberRange, Optional
from wtforms import DateField

class QuestionForm(FlaskForm):
    subject = StringField('제목', validators=[DataRequired('제목은 필수 입력 항목입니다.')])
    content = TextAreaField('내용', validators=[DataRequired('내용은 필수 입력 항목입니다.')])


class AnswerForm(FlaskForm):
    content = TextAreaField('내용', validators=[DataRequired('내용은 필수 입력 항목입니다.')])


class UserCreationForm(FlaskForm):
    username = StringField('아이디', validators=[DataRequired('아이디를 입력해주세요.'), Length(min=3, max=25)])
    password1 = PasswordField('비밀번호', validators=[ DataRequired('비밀번호를 입력해주세요.'), Length(min=4, max=100, message='비밀번호는 4자 이상 입력해주세요.'), EqualTo('password2', '비밀번호가 일치하지 않습니다')])
    password2 = PasswordField('비밀번호확인', validators=[DataRequired()])
    name = StringField('성함', validators=[DataRequired('성함은 필수 입력 항목입니다.')])
    phone = StringField('휴대폰 번호', validators=[  DataRequired('휴대폰 번호는 필수 입력 항목입니다.'),  Length(min=10, max=20, message='올바른 번호 형식이 아닙니다.') ])
    email = EmailField('이메일', validators=[Email()])
    address = StringField('주소')


class UserEditForm(FlaskForm):
    username = StringField( '아이디', validators=[  DataRequired('아이디를 입력해주세요.'),  Length(min=3, max=25) ])
    name = StringField('성함', validators=[DataRequired('성함은 필수 입력 항목입니다.')])
    phone = StringField('휴대폰 번호', validators=[ DataRequired('휴대폰 번호는 필수 입력 항목입니다.'), Length(min=10, max=20, message='올바른 번호 형식이 아닙니다.')])
    email = EmailField('이메일', validators=[Optional(), Email(message='올바른 이메일 형식이 아닙니다.')])
    address = StringField('주소')
    password1 = PasswordField('새 비밀번호',  validators=[Optional(), Length(min=4, max=100, message='비밀번호는 4자 이상 입력해주세요.'), EqualTo('password2', message='비밀번호가 일치하지 않습니다.')])
    password2 = PasswordField('새 비밀번호 확인', validators=[Optional()])


class FarmCreationForm(FlaskForm):
    farm_name = StringField('농가 이름', validators=[DataRequired('농가 이름은 필수 입력 항목입니다.'), Length(max=100)])
    region_l1 = SelectField('지역(시/도)', choices=[], validators=[DataRequired('시/도 정보는 필수입니다.')])
    region_l2 = SelectField('지역(시/군/구)', choices=[], validators=[DataRequired('시/군/구 정보는 필수입니다.')])
    total_area = FloatField('총 면적(㎡)', validators=[DataRequired('면적을 입력해주세요.'), NumberRange(min=0, message='면적은 0보다 커야 합니다.')])


class CultivationForm(FlaskForm):
    item = SelectField('품목', choices=[('완숙토마토', '완숙토마토')], validators=[DataRequired()])
    item_variety = StringField('품종')
    crop_cycle = SelectField('작기', choices=[(str(i), f"{i}작기") for i in range(1, 11)], validators=[DataRequired('작기를 선택해주세요')])
    planting_date = DateField('정식일', validators=[DataRequired()])
    planting_area = FloatField('식부면적(㎡)', validators=[DataRequired()])
    planting_density = FloatField('재식밀도')
    house_type = SelectField('온실종류', choices=[('', '선택'), ('비닐', '비닐'), ('유리', '유리'), ('기타', '기타')], validators=[DataRequired()])
    house_form = SelectField('온실유형', choices=[('', '선택'), ('연동', '연동'), ('단동', '단동'), ('광폭', '광폭'), ('연동양액', '연동양액'),('기타', '기타')], validators=[DataRequired()])




class UserLoginForm(FlaskForm):
    username = StringField('아이디', validators=[DataRequired('아이디를 입력해주세요.'), Length(min=3, max=25)])
    password = PasswordField('비밀번호', validators=[DataRequired('비밀번호를 입력해주세요.')])