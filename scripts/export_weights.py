"""Merge camera-head tensors and export a plain state_dict (no training state)."""
import argparse,hashlib,json
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--camera',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('infinihand_v1.pt'));a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 def state(path):
  x=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
  return x.get('trainable_state',x.get('state_dict',x))
 b=state(a.base);c=state(a.camera);prefix='backbone.model.camera_head.';keys={k for k in b if k.startswith(prefix)};overlay={k:v for k,v in c.items() if k.startswith(prefix)}
 if len(keys)!=69 or keys!=set(overlay):raise ValueError('Expected exactly the same 69 camera-head keys')
 merged=dict(b)
 for k,v in overlay.items():
  if v.shape!=b[k].shape:raise ValueError('Shape mismatch: '+k)
  merged[k]=v
 if not all(torch.is_tensor(v) for v in merged.values()):raise ValueError('Non-tensor state entry')
 a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp.pt');torch.save(merged,tmp)
 loaded=torch.load(tmp,map_location='cpu',mmap=True,weights_only=True)
 if set(loaded)!=set(merged):raise RuntimeError('Reload key mismatch')
 for k,v in loaded.items():
  if not torch.equal(v,merged[k]):raise RuntimeError('Reload tensor mismatch: '+k)
 tmp.rename(a.output);h=hashlib.sha256()
 with a.output.open('rb') as f:
  for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
 print(json.dumps(dict(output=str(a.output),tensors=len(loaded),camera_tensors=len(keys),bytes=a.output.stat().st_size,sha256=h.hexdigest()),indent=2))
if __name__=='__main__':main()
