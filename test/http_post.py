import requests
import json
import cv2
import base64

url = "http://localhost:8000/img_cls_base64"

def http_post_flask(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224, 224))
    img_bytes = cv2.imencode('.jpg', img_resized)[1].tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    data = {"img_base64": img_base64}
    res = requests.post(url, data=data)
    print(res.text)
    
if __name__ == '__main__':
    http_post_flask("test/cat.jpg")