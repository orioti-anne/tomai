from flask import Blueprint, url_for, render_template, g, flash
from werkzeug.utils import redirect
from datetime import datetime

from smartfarm import db
from ..forms import AnswerForm
from smartfarm.models import Question, Answer

bp = Blueprint('answer', __name__, url_prefix='/answer')


@bp.route('/create/<int:question_id>', methods=('POST',))
def create(question_id):
    if not g.user:
        flash('로그인 후 답변을 등록할 수 있습니다.')
        return redirect(url_for('auth.login'))

    form = AnswerForm()
    question = Question.query.get_or_404(question_id)

    if form.validate_on_submit():
        answer = Answer(
            content=form.content.data,
            created_date=datetime.now(),
            question_id=question.id,
            user_id=g.user.user_id
        )
        db.session.add(answer)
        db.session.commit()
        return redirect(url_for('question.detail', question_id=question_id))

    return render_template('question/question_detail.html', question=question, form=form)


@bp.route('/delete/<int:answer_id>/', methods=('POST',))
def delete(answer_id):
    if not g.user:
        flash('로그인 후 삭제할 수 있습니다.')
        return redirect(url_for('auth.login'))

    answer = Answer.query.get_or_404(answer_id)
    question_id = answer.question_id

    if answer.user_id != g.user.user_id:
        flash('본인이 작성한 답변만 삭제할 수 있습니다.')
        return redirect(url_for('question.detail', question_id=question_id))

    db.session.delete(answer)
    db.session.commit()
    flash('답변이 삭제되었습니다.')
    return redirect(url_for('question.detail', question_id=question_id))