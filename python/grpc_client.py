from __future__ import print_function

import logging

import grpc
import cv2
import base64
from protos.grpc_infer_service_pb2 import Request, Response
from protos.grpc_infer_service_pb2_grpc import InferenceServiceStub

def get_image_classify_result(address, *, img_path=None, img_base64=None):
    if img_base64 is None:
        if img_path is None:
            logging.error("img_path and img_base64 are both None")
            return ""
        img = cv2.imread(img_path)
        if img is None:
            logging.error("Failed to read image: {}".format(img_path))
            return
        resize_img = cv2.resize(img, (224, 224))
        # base64 encoding 
        img_bytes = cv2.imencode('.jpg', resize_img)[1].tobytes()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    assert type(img_base64) == str

    request = Request()
    request.name = img_base64
    
    with grpc.insecure_channel(address) as channel:
        stub = InferenceServiceStub(channel)
        response = stub.GetImgClsResult(request)
    # print("client received: " + response.message)
    logging.info("client received: " + response.message)
    return response.message

if __name__ == "__main__":
    address = "localhost:50051"
    img_path = "test/cat.jpg"
    import sys
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)d\t%(message)s', 
        datefmt="%H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
        ]
    )
    
    get_image_classify_result(address, img_path=img_path)
