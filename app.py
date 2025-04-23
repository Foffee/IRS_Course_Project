import os
from flask import Flask, request, jsonify

from collaborative_filtering import get_als_model
from recommend import recommend

from sanitize_json import sanitize_json

def create_app():
    app = Flask(__name__)

    # Only load model if this is the main process
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        get_als_model()  # Load once, reused from cache

    return app

app = create_app()

@app.route('/get-recommendations-for-you', methods=['POST'])
def get_recommendations():
    # Parse request body (JSON)
    req_body = request.get_json(force=True)

    # Extract guest data from the parsed body
    guest_data = req_body['guestData']

    # Obtain the personalized recommendations for the client
    recommendations = recommend( guest_data )

    # recommendations may have NaN's and other strange values to Javascript so they must be sanitized
    clean_recommendations = sanitize_json(recommendations)

    # Return personalized recommendations to client
    return jsonify({'success': True, 'recommendations': clean_recommendations})


if __name__ == '__main__':
    app.run(debug=True)
 