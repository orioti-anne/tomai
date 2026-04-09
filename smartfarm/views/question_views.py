from flask import Blueprint, render_template, url_for, request, g, flash
from werkzeug.utils import redirect
from ..forms import QuestionForm, AnswerForm
from datetime import datetime

from smartfarm.models import Question
from .. import db

bp = Blueprint('question', __name__, url_prefix='/question')


@bp.route('/list/')
def _list():
    page = request.args.get('page', type=int, default=1)
    question_list = Question.query.order_by(
        Question.created_date.desc(),
        Question.id.desc()
    )
    question_list = question_list.paginate(page=page, per_page=10)
    return render_template('question/question_list.html', question_list=question_list)


@bp.route('/detail/<int:question_id>/')
def detail(question_id):
    form = AnswerForm()
    question = Question.query.get_or_404(question_id)
    return render_template('question/question_detail.html', question=question, form=form)


@bp.route('/create/', methods=('GET', 'POST'))
def create():
    if not g.user:
        flash('로그인 후 질문을 등록할 수 있습니다.')
        return redirect(url_for('auth.login'))

    form = QuestionForm()

    if request.method == 'POST' and form.validate_on_submit():
        question = Question(
            subject=form.subject.data,
            content=form.content.data,
            created_date=datetime.now(),
            user_id=g.user.user_id
        )
        db.session.add(question)
        db.session.commit()
        return redirect(url_for('question._list'))

    return render_template('question/question_form.html', form=form)


@bp.route('/delete/<int:question_id>/', methods=('POST',))
def delete(question_id):
    if not g.user:
        flash('로그인 후 삭제할 수 있습니다.')
        return redirect(url_for('auth.login'))

    question = Question.query.get_or_404(question_id)

    if question.user_id != g.user.user_id:
        flash('본인이 작성한 질문만 삭제할 수 있습니다.')
        return redirect(url_for('question.detail', question_id=question_id))

    db.session.delete(question)
    db.session.commit()
    flash('질문이 삭제되었습니다.')
    return redirect(url_for('question._list'))