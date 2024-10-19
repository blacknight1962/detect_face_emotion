from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import base64
from deepface import DeepFace
import os
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

if not os.path.exists('saved_faces'):
    os.makedirs('saved_faces')  # 저장할 디렉토리 생성

app = Flask(__name__)

# 모델을 전역 변수로 캐시
model = DeepFace.build_model('VGG-Face')  # 모델 로드

@app.route('/')
def home():
    return render_template('index.html')  # HTML 파일 렌더링
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('saved_faces', filename)
        file.save(file_path)  # 파일 저장
        
        # 얼굴 인식 처리
        result = DeepFace.analyze(file_path, actions=['emotion'], enforce_detection=False)
        
        # 결과를 JSON 형식으로 반환
        return jsonify(result)

    return jsonify({'error': 'File upload failed'})

@app.route('/upload', methods=['POST'])
def analyze():
    # 클라이언트에서 전송한 이미지 데이터 받기
    data = request.json['image']
    
    # Base64로 인코딩된 이미지를 디코딩
    img_data = base64.b64decode(data.split(',')[1])
    np_img = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # 얼굴 인식 및 감정 분석, 나이 추정
    try:
        # 감정 및 나이 분석
        result = DeepFace.analyze(img, actions=['emotion', 'age'], enforce_detection=False)
        print(result)  # 결과를 콘솔에 출력하여 확인

        # 얼굴 비교
        face_found = False
        recognized_name = "???"  # 기본값은 '???'
        
        # 저장된 얼굴과 비교
        for filename in os.listdir('saved_faces'):
            # 저장된 얼굴 이미지 로드
            saved_face_path = os.path.join('saved_faces', filename)
            saved_face = cv2.imread(saved_face_path)

            # 얼굴 비교
            try:
                comparison_result = DeepFace.verify(img, saved_face, model_name='VGG-Face', enforce_detection=False)
                if comparison_result['verified']:
                    face_found = True
                    # 메타데이터에서 이름 가져오기
                    id = filename.split('_')[1].split('.')[0]  # 파일명에서 ID 추출
                    recognized_name = get_name_from_metadata(id)  # 메타데이터에서 이름 가져오기
                    break
            except Exception as e:
                print(f"Error comparing with {filename}: {str(e)}")

        # 결과에 이름 추가
        if isinstance(result, list):
            for res in result:
                res['name'] = recognized_name if face_found else None  # 얼굴이 발견되면 이름 추가
        else:
            result['name'] = recognized_name if face_found else None  # 단일 객체일 경우 처리

        return jsonify(result)  # 분석 결���를 JSON 형식으로 반환
    except Exception as e:
        print(f"Error: {str(e)}")  # 오류 메시지를 콘솔에 출력
        return jsonify({"error": str(e)}), 400  # 오류 발생 시 에러 메시지 반환

@app.route('/upload_face', methods=['POST'])
def upload_face():
    file = request.files['image']
    filename = secure_filename(file.filename)
    file.save(os.path.join('saved_faces', filename))  # 저장할 디렉토리 지정
    return jsonify({"message": "얼굴 이미지가 저장되었습니다."})

@app.route('/register', methods=['POST'])
def register():
    data = request.json  # JSON 데이터 수신
    print(data)  # 수신된 데이터 로그 출력 (디버깅 용도)

    # 데이터에서 no, id, name 추출
    try:
        no = data['no']
        id = data['id']
        name = data['name']
        image_data = data['image']
    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400  # 키가 없을 경우 에러 반환

    # 이미지 데이터 처리
    img_data = base64.b64decode(image_data.split(',')[1])  # Base64로 인코딩된 이미지를 디코딩
    np_img = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Failed to decode image"}), 400  # 이미지 디코딩 실패 시 에러 환

    # 파일명 생성
    filename = f"{no}_{id}.png"  # 이름에서 한글 제거
    save_path = os.path.join('saved_faces', filename)  # 저장할 경로

    # 이미지 저장
    success = cv2.imwrite(save_path, img)  # 이미지를 지정된 경로에 저장
    print(f"Image saved at: {save_path}")  # 저장된 파일 경로 출력

    if success:
        # 메타데이터 파일 생성 및 업데이트
        metadata_file = 'metadata.json'
        metadata = {}

        # 기존 메타데이터 로드
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as meta_file:
                metadata = json.load(meta_file)

        # 메타데이터 업데이트
        metadata[id] = name

        # 메타데이터 파일 저장
        with open(metadata_file, 'w', encoding='utf-8') as meta_file:
            json.dump(metadata, meta_file, ensure_ascii=False, indent=4)

        return jsonify({"message": "Image saved successfully", "filename": filename})
    else:
        print(f"Failed to save image at: {save_path}")  # 저장 실패 로그
        return jsonify({"error": "Failed to save image"}), 500  # 저장 실패 시 에러 반환

def get_name_from_metadata(id):
    metadata_file = 'metadata.json'
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as meta_file:
            metadata = json.load(meta_file)
            return metadata.get(id, "???")  # ID에 해당하는 이름 반환, 없으면 '???'
    return "???"

@app.route('/search', methods=['POST'])
def search_face():
    # 클라이언트에서 전송한 이미지 데이터 받기
    data = request.json['image']
    
    # Base64로 인코딩된 이미지를 디코딩
    img_data = base64.b64decode(data.split(',')[1])
    np_img = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # 얼굴 비교
    face_found = False
    recognized_name = "???"  # 기본값은 '???'
    
    # 저장된 얼굴과 비교
    for filename in os.listdir('saved_faces'):
        # 저장된 얼굴 이미지 로드
        saved_face_path = os.path.join('saved_faces', filename)
        saved_face = cv2.imread(saved_face_path)

        # 얼굴 비교
        try:
            comparison_result = DeepFace.verify(img, saved_face, model_name='VGG-Face', enforce_detection=False)
            if comparison_result['verified']:
                face_found = True
                # 메타데이터에서 이름 가져오기
                id = filename.split('_')[1].split('.')[0]  # 파일명에서 ID 추출
                recognized_name = get_name_from_metadata(id)  # 메타데이터에서 이름 가져오기
                break
        except Exception as e:
            print(f"Error comparing with {filename}: {str(e)}")

    return jsonify({"name": recognized_name if face_found else None})  # 검색 결과 반환

if __name__ == '__main__':
    app.run(debug=True)
