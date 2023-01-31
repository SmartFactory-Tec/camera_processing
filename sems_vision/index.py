from flask import Blueprint, current_app, render_template

bp = Blueprint('index', __name__)


@bp.route('/')
def index_route():
    """Video streaming home page."""
    return render_template('indexv4.html', len=len(current_app.config['camera_ids']), camaraIDs=current_app.config['camera_ids'])
