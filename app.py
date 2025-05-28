# app.py

from flask import Flask, render_template, jsonify, request
import fast_ai_player

app = Flask(__name__)

game_generator = fast_ai_player.get_ai_game_frame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_frame')
def get_frame():
    global game_generator
    try:
        frame = next(game_generator)
        return jsonify(frame)
    except StopIteration:
        return jsonify({'done': True})

@app.route('/set_model', methods=['POST'])
def set_ai_model():
    data = request.get_json()
    model_name = data.get('model', 'best').lower()
    mode = data.get('mode', 'normal')

    model_map = {
        'early': 'model_episode_0.pth',
        'mid': 'model_episode_100.pth',
        'late': 'model_episode_250.pth',
        'best': 'best_dqn_snake.pth',
        'final': 'final_dqn_snake.pth'
    }

    model_path = model_map.get(model_name, 'best_dqn_snake.pth')

    fast_ai_player.set_model(model_path=model_path, mode=mode)
    return jsonify({"status": "success", "selected": model_name, "mode": mode})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)