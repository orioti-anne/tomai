from flask import Blueprint, render_template, url_for, request
from smartfarm.models import Question
from werkzeug.utils import redirect
from ..forms import QuestionForm, AnswerForm
from datetime import datetime
from .. import db

bp = Blueprint('question', __name__, url_prefix='/question')


@bp.route('/list/')
def _list():
    page = request.args.get('page', type=int, default=1)
    question_list = Question.query.order_by(Question.created_date.desc(), Question.id.desc())
    question_list = question_list.paginate(page=page, per_page=10)
    return render_template('question/question_list.html', question_list=question_list)

@bp.route('/detail/<int:question_id>/')
def detail(question_id):
    form = AnswerForm()
    question = Question.query.get_or_404(question_id)
    return render_template('question/question_detail.html', question=question, form=form)

@bp.route('/create/', methods=('GET', 'POST'))
def create():
    form = QuestionForm()

    if request.method == 'POST' and form.validate_on_submit():  # 성공(1.제목/내용 입력 2.CSRF 보안코튼 일치 3.기타 규칙 통과) 시 redirect, 실패 시 forward 호출
        question = Question(
            subject=form.subject.data,
            content=form.content.data,
            created_date=datetime.now()
        )
        db.session.add(question)
        db.session.commit()
        return redirect(url_for('main.index'))

    return render_template('question/question_form.html', form=form)    # render_template() 호출 행위가 forward()와 유사한 역할