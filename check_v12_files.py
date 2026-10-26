"""Offline checks against the supplied v12 model and backend data (Keras zip format)."""
import json
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
names=json.loads((ROOT/'model/class_names.json').read_text(encoding='utf8'))
assert len(names)==len(set(names))==12
assert 'Long Pepper' not in names
taxonomy=json.loads((ROOT/'taxonomy_mapping.json').read_text(encoding='utf8'))
assert set(taxonomy)==set(names)
model=ROOT/'model/leafvision_model_v12.keras'
with zipfile.ZipFile(model) as z:
    cfg=json.loads(z.read('config.json'))
    layers=cfg['config']['layers']
    assert any(layer['class_name']=='InputLayer' and layer['config']['batch_shape']==[None,224,224,3] for layer in layers)
    assert any(layer['class_name']=='Dense' and layer['config'].get('units')==12 for layer in layers)
    assert any(layer['class_name']=='Rescaling' and abs(layer['config'].get('scale',0)-(1/127.5))<1e-9 for layer in layers)
print('PASS: class mapping, taxonomy coverage, model IO and embedded rescaling')
