from flask_wtf import FlaskForm
from wtforms.fields.simple import StringField, TextAreaField, PasswordField
from wtforms.fields import EmailField
from wtforms.validators import DataRequired, Length, Email, EqualTo


class QuestionForm(FlaskForm):
    subject = StringField('제목', validators=[DataRequired('제목은 필수 입력 항목입니다.')])
    content = TextAreaField('내용', validators=[DataRequired('내용은 필수 입력 항목입니다.')])


class AnswerForm(FlaskForm):
    content = TextAreaField('내용', validators=[DataRequired('내용은 필수 입력 항목입니다')])


class UserCreationForm(FlaskForm):
    username = StringField('사용자이름', validators=[DataRequired(), Length(min=3, max=25)])
    password1 = PasswordField('비밀번호', validators=[
        DataRequired(), EqualTo('password2', '비밀번호가 일치하지 않습니다')])
    password2 = PasswordField('비밀번호확인', validators=[DataRequired()])
    name = StringField('성함', validators=[DataRequired('성함은 필수 입력 항목입니다.')])
    phone = StringField('휴대폰 번호', validators=[
        DataRequired('휴대폰 번호는 필수 입력 항목입니다.'),
        Length(min=10, max=20, message='올바른 번호 형식이 아닙니다.')
    ])
    email = EmailField('이메일', validators=[Email()])
    address = StringField('주소')


class UserLoginForm(FlaskForm):
    username = StringField('사용자이름', validators=[DataRequired(), Length(min=3, max=25)])
    password = PasswordField('비밀번호', validators=[DataRequired()])