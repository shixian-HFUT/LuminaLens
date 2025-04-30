from flask import Flask, request, render_template, jsonify, Response, url_for
from werkzeug.utils import secure_filename
import os
from predict import predict
from openai import OpenAI
import base64
import pickle # <-- Add pickle import
import sys # <-- Add sys import
import numpy as np # <-- 确保导入 numpy
import shutil # <-- 导入 shutil 用于删除目录内容

# --- Add the tools directory to sys.path --- 
# This ensures the import works correctly regardless of where the script is run from
tools_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tools', 'Resnet50'))
if tools_dir not in sys.path:
    sys.path.insert(0, tools_dir)

# --- Import the ResNet50 search function --- 
try:
    # 确保从 tools_dir 导入
    from search_ResNet50 import find_similar_images #, extract_features # extract_features 可能不需要在 app.py 中直接使用
except ImportError as e:
    print(f"Error importing from search_ResNet50: {e}")
    def find_similar_images(*args, **kwargs):
        print("Warning: search_ResNet50.py not found or import failed. Similarity search disabled.")
        return []

# --- Remove the old import --- 
# from similarity_search_tags import search_similar_images_with_text_and_tags # <-- Remove or comment out this line

app = Flask(__name__)

# 上传路径配置
UPLOAD_FOLDER = 'static/uploads'
SIMILAR_FOLDER = 'static/similar_images' # <-- Define similar images folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SIMILAR_FOLDER, exist_ok=True) # <-- Create similar images folder if not exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SIMILAR_FOLDER'] = SIMILAR_FOLDER # <-- Store similar folder path

# --- Load metadata and tags (outside the request for efficiency if possible, or handle errors) ---
# --- Load metadata and tags using absolute paths relative to tools_dir --- 
METADATA_PATH = os.path.join(tools_dir, 'metadata.pkl')
TAGS_PATH = os.path.join(tools_dir, 'tags.txt')

metadata = {}
tag_map = {}
try:
    # 使用构建好的绝对路径加载文件
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    with open(TAGS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) < 2:
                continue
            tag_id = int(parts[0])
            tag_name = parts[1]
            tag_map[tag_id] = tag_name
except FileNotFoundError:
    print(f"Warning: {METADATA_PATH} or {TAGS_PATH} not found. Scores and tags will not be available.")
except Exception as e:
    print(f"Error loading metadata or tags: {e}")

# 通义千问客户端配置
client = OpenAI(
    api_key="sk-1c5c4fb2b35646b5bba7491c1acd05ff",  # 替换为你的通义千问API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 添加Base64编码函数
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

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

    # --- 清空旧的相似图片 --- 
    similar_dir = app.config['SIMILAR_FOLDER']
    if os.path.exists(similar_dir):
        for item in os.listdir(similar_dir):
            item_path = os.path.join(similar_dir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"无法删除 {item_path}. 原因: {e}")
    else:
        os.makedirs(similar_dir) # 如果目录不存在，则创建

    # 保存图片到本地 (使用绝对路径)
    absolute_image_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    try:
        image.save(absolute_image_path)
        if not os.path.exists(absolute_image_path):
            # 添加更详细的日志
            print(f"Error: Failed to save image to {absolute_image_path}")
            return jsonify({'error': f"保存失败，路径：{absolute_image_path}"}), 500
    except Exception as e:
         # 添加更详细的日志
         import traceback
         print(f"Error saving image: {str(e)}")
         print(traceback.format_exc())
         return jsonify({'error': f"保存图片时出错: {str(e)}"}), 500

    # --- 生成图片的 URL (使用正斜杠拼接) ---
    upload_folder_name = os.path.basename(app.config['UPLOAD_FOLDER'])
    # 先替换反斜杠，再用于 f-string
    safe_filename = filename.replace('\\', '/')
    image_url = url_for('static', filename=f"{upload_folder_name}/{safe_filename}")

    # 生成Base64编码
    base64_image = None # 初始化为 None
    try:
        base64_image = encode_image(absolute_image_path)
    except Exception as e:
        # 记录错误，但允许流程继续
        print(f"图片编码失败：{str(e)}")
        # 不在此处返回错误，让后续步骤（如评分和相似性搜索）继续

    # --- Get score first --- 
    score = None
    try:
        score = predict(absolute_image_path)
    except Exception as e:
        print(f"图片评分失败: {str(e)}")
        # Decide if you want to return an error or continue without score
        # return jsonify({'error': f'图片评分失败: {str(e)}'}), 500

    ai_analysis_result = "AI 分析失败或未执行" # 默认值
    if base64_image and score is not None: # <-- Check if score is available
        try:
            # Call analyze_image_stream with the score
            ai_analysis_result = analyze_image_stream(base64_image, score) # <-- Pass score here
        except Exception as e:
            print(f"AI 分析调用失败: {str(e)}")
            # 保留默认错误消息
    elif not base64_image:
        print("由于图片编码失败，跳过 AI 分析。")
    elif score is None:
        print("由于评分失败，跳过 AI 分析。")

    try:
        # --- ResNet50 相似度检索 --- 
        save_dir_absolute = os.path.abspath(app.config['SIMILAR_FOLDER'])
        num_candidates_to_find = 10
        
        similar_image_paths_saved_absolute = find_similar_images(
            query_image_path=absolute_image_path,
            top_k=num_candidates_to_find,
            save_to_dir=save_dir_absolute
        )

        # --- Process results: Get scores, tags, sort, and generate URLs ---
        processed_similar_images = []
        similar_folder_name = os.path.basename(app.config['SIMILAR_FOLDER'])
        for saved_abs_path in similar_image_paths_saved_absolute:
            original_filename = os.path.basename(saved_abs_path)
            # 先替换反斜杠，再用于 f-string
            safe_original_filename = original_filename.replace('\\', '/')
            similar_image_url = url_for('static', filename=f"{similar_folder_name}/{safe_original_filename}")
            
            img_id, _ = os.path.splitext(original_filename)

            # 使用已加载的 metadata 和 tag_map
            avg_score = float('nan') # Default score
            tags = ['元数据未找到'] # Default tags
            if img_id in metadata:
                meta = metadata[img_id]
                avg_score = meta.get("avg_score", float('nan'))
                tag_ids = meta.get("tags", [])
                tags = [tag_map.get(t, f'未知:{t}') for t in tag_ids]
            
            # 确保 avg_score 是数字或 NaN
            if not isinstance(avg_score, (int, float)):
                avg_score = float('nan')

            processed_similar_images.append({
                "path": similar_image_url, # <-- Use the correctly generated URL
                "score": avg_score,
                "tags": tags
            })

        # Sort by average score (descending), handling NaN scores using numpy
        processed_similar_images.sort(key=lambda x: x['score'] if not np.isnan(x['score']) else -np.inf, reverse=True)

        top_2_similar_images = processed_similar_images[:2]

        # 返回JSON响应 (确保包含 ai_analysis_result)
        response = {
            'score': score, # <-- Use the calculated score
            'ai_analysis': ai_analysis_result, # <-- 确保 AI 分析结果包含在内
            'original_image': image_url, 
            'similar_images': top_2_similar_images 
        }
        print(f"--- 准备返回给前端的 JSON 数据: {response} ---") # 新增日志
        return jsonify(response)

    except Exception as e:
        import traceback
        print(f"处理图片或相似度搜索时出错: {str(e)}")
        print(traceback.format_exc())
        # 返回包含错误的 JSON，但确保 AI 分析结果（即使是错误消息）也包含在内
        error_response = {
            'score': score, # Include score if available, else None
            'ai_analysis': ai_analysis_result, # 包含之前的分析结果或错误信息
            'original_image': image_url, # <-- 同样修改此处的键名
            'similar_images': [],
            'error': f'处理图片时发生错误: {str(e)}'
        }
        print(f"--- 准备返回给前端的错误 JSON 数据: {error_response} ---") # 新增日志
        return jsonify(error_response), 500

@app.route('/similar_images/<path:filename>')
def similar_images(filename):
    return url_for('static', filename=f'similar_images/{filename}')

def analyze_image_stream(base64_image, score): # <-- Add score parameter
    print(f"--- 开始调用 AI 分析 (Score: {score}) ---") # Log score
    try:
        # 根据分数确定质量等级
        if score < 4.5:
            quality_level = "低分"
        elif 4.5 <= score <= 5.5:
            quality_level = "中等评分"
        else:
            quality_level = "高分"
        score_info = f"这张图片的质量评分为 {score:.2f} ({quality_level})。"

        completion = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": score_info}, # <-- 这里加入了评分高低信息
                        {"type": "text", "text": "这是一张摄影小白的图片，你是一名摄影技巧高超的摄影师，请结合图片评分，详细逐条分析这张图片的构图、色彩和摄影技巧，指出其优点与不足并提出改进建议。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}" # 使用Base64编码
                            }
                        }
                    ]
                }
            ],
            stream=True
        )
        content = "" # <-- This is line 296, syntax is correct here
        print("--- AI 分析流式响应开始 --- ") # 新增日志
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    # print(f"Chunk content: {delta.content}") # 可以取消注释以查看每个块
                    content += delta.content
        print(f"--- AI 分析流式响应结束，总内容长度: {len(content)} ---") # 新增日志
        if not content:
             print("--- 警告: AI 分析返回内容为空 --- ") # 新增日志
        return content
    except Exception as e:
        import traceback # 确保导入 traceback
        print(f"--- AI 分析 API 调用失败：{str(e)} ---") # 修改日志格式
        print(traceback.format_exc()) # 打印详细堆栈
        return f"API 调用失败：{str(e)}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
