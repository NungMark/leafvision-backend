"""Minimal authenticated production smoke test; credentials are environment variables.
Run: python smoke_test.py https://leafvision-backend.onrender.com sample_leaf.jpg
Set LEAFVISION_LOGIN and LEAFVISION_PASSWORD locally; do not share tokens.
"""
import json, os, sys, mimetypes
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from uuid import uuid4

BASE = (sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8000').rstrip('/')
IMAGE = sys.argv[2] if len(sys.argv)>2 else None

def request(path, method='GET', payload=None, headers=None):
    req=Request(BASE+path, data=payload, headers=headers or {}, method=method)
    with urlopen(req,timeout=90) as response:
        return response.status, json.load(response)

status, health = request('/health')
assert status == 200 and health.get('model_loaded') and health.get('model_version')=='v12' and health.get('number_of_classes')==12, health
print('PASS /health model v12/12 classes, Grad-CAM available:',health.get('gradcam_available'))

user=os.getenv('LEAFVISION_LOGIN','')
password=os.getenv('LEAFVISION_PASSWORD','')
if not user or not password:
    print('SKIP authenticated checks: set LEAFVISION_LOGIN, LEAFVISION_PASSWORD as environment variables')
    sys.exit(0)
login_data=json.dumps({'identifier':user,'password':password}).encode()
status, login=request('/auth/login','POST',login_data,{'Content-Type':'application/json'})
assert status==200 and login.get('access_token'), login
hdr={'Authorization':'Bearer '+login['access_token']}
status, me=request('/auth/me',headers=hdr)
assert status==200, me
print('PASS /auth/login and /auth/me')
if not IMAGE:
    print('SKIP predict; provide an image path')
    sys.exit(0)

def upload(path):
    boundary='----LeafVision'+uuid4().hex
    raw=open(IMAGE,'rb').read()
    data=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{os.path.basename(IMAGE)}"\r\nContent-Type: image/jpeg\r\n\r\n').encode()+raw+(f'\r\n--{boundary}--\r\n').encode()
    status,result=request(path,'POST',data,{**hdr,'Content-Type':f'multipart/form-data; boundary={boundary}'})
    assert status==200, result
    return result
pred=upload('/predict')
assert pred.get('model_version')=='v12' and len(pred.get('top3',[]))==3,pred
print('PASS /predict',pred.get('prediction'),pred.get('confidence'))
if health.get('gradcam_available'):
    cam=upload('/explain/gradcam')
    assert cam.get('model_version')=='v12' and cam.get('overlay_base64'),cam
    # /predict may return Unknown when the confidence is below 60%.
    assert cam['class_name']==pred['top3'][0]['name'],(cam['class_name'],pred['top3'][0]['name'])
    print('PASS /explain/gradcam',cam['class_name'],cam.get('confidence'))
