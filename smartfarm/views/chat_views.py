import requests
from flask import Blueprint, render_template, request, jsonify, make_response

bp = Blueprint('chat', __name__)

TOMAI_API = "http://localhost:8003/chat"


@bp.route('/chat')
def chat_page():
    return render_template('chat.html')


@bp.route('/slm')
def slm_page():
    resp = make_response(render_template("slm.html"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


SLM_LOCAL = "http://localhost:8004"


@bp.route('/slm/upload-doc', methods=['POST'])
def slm_upload_doc():
    f = request.files['file']
    resp = requests.post(
        SLM_LOCAL + '/upload-doc',
        files={'file': (f.filename, f.stream, f.content_type)},
        timeout=60
    )
    return resp.content, resp.status_code, {'Content-Type': 'application/json'}


@bp.route('/slm/fetch-doc', methods=['POST'])
def slm_fetch_doc():
    resp = requests.post(SLM_LOCAL + '/fetch-doc', json=request.get_json(), timeout=30)
    return resp.content, resp.status_code, {'Content-Type': 'application/json'}


@bp.route('/slm/doc/<doc_id>', methods=['DELETE'])
def slm_delete_doc(doc_id):
    resp = requests.delete(SLM_LOCAL + f'/doc/{doc_id}', timeout=10)
    return resp.content, resp.status_code, {'Content-Type': 'application/json'}


@bp.route('/slm/chat/stream')
def slm_chat_stream():
    from flask import stream_with_context, Response
    params = request.args.to_dict()
    upstream = requests.get(
        SLM_LOCAL + '/chat/stream',
        params=params,
        stream=True,
        timeout=None
    )
    return Response(
        stream_with_context(upstream.iter_content(chunk_size=1)),
        content_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@bp.route('/api/chat-pure', methods=['POST'])
def chat_pure():
    data = request.get_json()
    message = (data or {}).get('message', '').strip()
    history = (data or {}).get('history', [])
    if not message:
        return jsonify({'reply': ''}), 400
    try:
        resp = requests.post(TOMAI_API, json={'message': message, 'history': history}, timeout=60)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.ConnectionError:
        return jsonify({'reply': '⚠️ AI 서버에 연결할 수 없습니다.'}), 503
    except requests.Timeout:
        return jsonify({'reply': '⚠️ 응답 시간이 초과되었습니다.'}), 504
    except Exception as e:
        return jsonify({'reply': f'⚠️ 오류: {str(e)}'}), 500
