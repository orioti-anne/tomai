from flask import Blueprint, url_for, request, render_template
from werkzeug.utils import redirect
from datetime import datetime

from smartfarm import db
from ..forms import AnswerForm
from smartfarm.models import Question, Answer

bp = Blueprint('answer', __name__, url_prefix='/answer')

@bp.route('/create/<int:question_id>', methods=('GET', 'POST'))
def create(question_id):
    form=AnswerForm()
    question = Question.query.get_or_404(question_id)

    if form.validate_on_submit():
        content = request.form['content']
        answer = Answer(content=content, created_date=datetime.now())
        question.answers_set.append(answer)
        db.session.commit()
        return redirect(url_for('question.detail', question_id=question_id))

    return render_template('question/question_detail.html', question=question, form=form)





