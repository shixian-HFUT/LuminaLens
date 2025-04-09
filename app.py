from flask import Flask, request, render_template, jsonify, Response, url_for
from werkzeug.utils import secure_filename
import os
from predict import predict
from openai import OpenAI

app = Flask(__name__)

# 上传路径配置
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 通义千问客户端配置
client = OpenAI(
    api_key="sk-1c5c4fb2b35646b5bba7491c1acd05ff",  # 替换为你的通义千问API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def analyze_image_stream(image_url):
    try:
        completion = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请详细分析这张图片的构图、色彩和摄影技巧，并提出改进建议。"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            stream=True
        )

        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content"):
                    yield delta.content
                else:
                    # 处理其他可能的字段（如 role、content 等）
                    print(f"忽略非 content 字段：{delta}")
    except Exception as e:
        yield f"API 调用失败：{str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': '未选择图片'}), 400

    image = request.files['image']
    filename = secure_filename(image.filename)
    if not filename:
        return jsonify({'error': '无效的文件名'}), 400

    # 保存图片到本地，强制使用正斜杠路径
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace("\\", "/")
    image.save(image_path)
    if not os.path.exists(image_path):
        return jsonify({'error': f"保存失败，路径：{image_path}"}), 500

    # 生成静态文件URL
    image_url = url_for('static', filename=f"uploads/{filename}", _external=True)
    print(f"原始 image_url: {image_url}")  # 调试输出

    # 替换为 ngrok 的公网地址（注意末尾斜杠）
    ngrok_url = "https://b2ca-36-161-107-179.ngrok-free.app/"# 末尾添加斜杠
    image_url = image_url.replace(request.host_url, ngrok_url)
    print(f"替换后的 image_url: {image_url}")  # 调试输出

    try:
        score = predict(image_path)

        def stream():
            yield f"图片评分：{score}/10\n\nAI分析结果：\n"
            for content in analyze_image_stream(image_url):
                yield content

        return Response(stream(), content_type='text/plain;charset=utf-8')

    except Exception as e:
        return jsonify({'error': f"处理失败：{str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)