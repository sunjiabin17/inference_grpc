from flask import Flask, Blueprint
from flask import request
import base64

import threading
from grpc_client import get_image_classify_result
app = Flask(__name__)

# # 定义一个 Blueprint，包含一个路由和视图函数
# bp_server = Blueprint('infer_service', __name__)
# @bp_server.route('/hello')
# def hello():
#     return 'Hello from port 8000!'

# @bp_server.route('/img_cls')
# def img_cls(img_path):
#     # address = "localhost:50051"
#     # get_image_classify_result(address, img_path)
#     return 'img_cls: ' + img_path
    
@app.route('/hello')
def hello():
    return 'Hello from port 8000!'

@app.route('/img_cls')
def img_classify():
    img_path = request.args.get('img_path')
    address = "localhost:50051"
    res = get_image_classify_result(address, img_path=img_path)
    return res

# 定义启动 Web 服务器的函数
def run_server(port):
    app.run(port=port)

# 在 app 中注册两个 Blueprint，并启动两个 Web 服务器，分别监听 8000 和 9000 端口
if __name__ == '__main__':
    # app.register_blueprint(bp_server, url_prefix='')
    t1 = threading.Thread(target=run_server, args=(8000,))
    t1.start()
