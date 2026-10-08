"""Isolated fixed-keyframe backend ablation. No motion filtering or filler."""
from pathlib import Path
import sys, json, time
import cv2, numpy as np, torch
from scipy.spatial.transform import Rotation, Slerp

R=Path(__file__).resolve().parents[1]
D=R/'third_party/droid'
sys.path.insert(0,str(D/'droid_slam'))
from droid import Droid
from factor_graph import FactorGraph
import lietorch
from types import SimpleNamespace

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--intrinsics',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--stride',type=int,default=20)
    args_cli=parser.parse_args();out=args_cli.output;stride=args_cli.stride
    if stride<1:raise ValueError('stride must be positive')
    scene=out.name;rep=0
    torch.set_grad_enabled(False);torch.manual_seed(0);np.random.seed(0);cv2.setNumThreads(4)
    prep=time.perf_counter()
    with np.load(out/'ba_inputs.npz') as src:
        T=src['poses'].astype(np.float64);boxes=src['boxes']
    n=len(T);ids=np.unique(np.r_[np.arange(0,n,stride),n-1]).astype(int);count=len(ids)
    if count<2:raise ValueError('BA requires at least two keyframes')
    images={int(i):cv2.imread(str(out/'frames'/f'{i:06d}.png')) for i in ids}
    H,W=images[0].shape[:2];h,w=384,536
    calib=json.loads(args_cli.intrinsics.read_text())
    intr=torch.tensor([calib['fx']*w/W,calib['fy']*h/H,calib['cx']*w/W,calib['cy']*h/H],device='cuda')
    frames=[];masks=[]
    for t in ids:
        small=cv2.resize(images[t],(w,h));gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        valid=((gray>3).astype(np.uint8)*255)
        for b in boxes[t]:
            if b[2]<=b[0] or b[3]<=b[1]:continue
            x0,y0,x1,y1=(b*np.array([w,h,w,h])).astype(int)
            cv2.rectangle(valid,(max(0,x0-20),max(0,y0-20)),(min(w-1,x1+20),min(h-1,y1+20)),0,-1)
        frames.append(torch.from_numpy(small).permute(2,0,1)[None])
        masks.append(torch.from_numpy((valid[3::8,3::8]==0).astype(np.float32)).cuda())
    prep_seconds=time.perf_counter()-prep
    start=time.perf_counter()
    args=SimpleNamespace(weights=str(args_cli.weights),buffer=count+1,
        image_size=[h,w],disable_vis=True,stereo=False,beta=.3,filter_thresh=2.4,warmup=8,
        keyframe_thresh=4.,frontend_thresh=16.,frontend_window=25,frontend_radius=2,frontend_nms=1,
        backend_thresh=22.,backend_radius=2,backend_nms=3,upsample=False)
    system=Droid(args);v=system.video;f=system.filterx
    del system.frontend
    system.traj_filler=None
    torch.cuda.synchronize();load_seconds=time.perf_counter()-start
    w2c=np.linalg.inv(T[ids]);seed=torch.tensor(np.c_[w2c[:,:3,3],Rotation.from_matrix(w2c[:,:3,:3]).as_quat()],device='cuda',dtype=torch.float32)
    torch.cuda.synchronize();start=time.perf_counter()
    for j,(stamp,frame,mask) in enumerate(zip(ids,frames,masks)):
        inputs=frame[None,:,[2,1,0]].cuda()/255.0
        inputs=inputs.sub_(f.MEAN).div_(f.STDV)
        gmap=f._MotionFilter__feature_encoder(inputs)
        net,inp=f._MotionFilter__context_encoder(inputs[:,[0]])
        v.append(int(stamp),frame[0],seed[j],1.0,None,intr/8.0,gmap,net[0],inp[0],mask)
    torch.cuda.synchronize();feature_seconds=time.perf_counter()-start
    assert v.counter.value==count and torch.equal(v.poses[:count],seed)
    assert torch.count_nonzero(v.disps_sens)==0
    scale=[1.];normalizations=[];ba_calls=[0];graphs=[]
    original_norm=v.normalize;original_ba=v.ba;original_graph=FactorGraph.add_proximity_factors
    def norm():
        fac=float(v.disps[:v.counter.value].mean());original_norm();scale[0]*=fac;normalizations.append(fac)
    def ba(*a,**kw):ba_calls[0]+=1;return original_ba(*a,**kw)
    def graph(self,*a,**kw):
        ret=original_graph(self,*a,**kw);graphs.append({'vertices':v.counter.value,'edges':len(self.ii)});return ret
    v.normalize=norm;v.ba=ba;FactorGraph.add_proximity_factors=graph
    torch.cuda.synchronize();start=time.perf_counter();stages=[]
    for iterations in [7,12]:
        torch.cuda.empty_cache();torch.cuda.synchronize();t=time.perf_counter()
        system.backend(iterations);torch.cuda.synchronize()
        stages.append({'stage':'backend','iterations':iterations,'seconds':time.perf_counter()-t})
    backend_seconds=time.perf_counter()-start
    start=time.perf_counter()
    keyposes=lietorch.SE3(v.poses[:count]).inv().matrix().cpu().numpy().astype(np.float64)
    keyposes[:,:3,3]/=scale[0]
    keyposes=np.linalg.inv(keyposes[0])@keyposes
    corrections=keyposes@np.linalg.inv(T[ids])
    delta=np.tile(np.eye(4),(n,1,1))
    delta[:,:3,:3]=Slerp(ids,Rotation.from_matrix(corrections[:,:3,:3]))(np.arange(n)).as_matrix()
    for axis in range(3):delta[:,axis,3]=np.interp(np.arange(n),ids,corrections[:,axis,3])
    Q=delta@T
    interp_seconds=time.perf_counter()-start
    assert np.allclose(Q[ids],keyposes,atol=2e-6) and np.allclose(Q[0],np.eye(4),atol=2e-6)
    assert np.isfinite(Q).all() and ba_calls[0]==19
    np.savez_compressed(out/'trajectory.npz',poses=Q,initial_poses=T,keyframe_ids=ids,keyframe_poses=keyposes,corrections=delta)
    config={'scene':scene,'frames':n,'keyframes':count,'stride':stride,'keyframe_ids':ids.tolist(),
        'external_depth':False,'frontend_BA':False,'motion_filter':False,'filler':False,
        'interpolation':'SLERP rotation and linear translation of world-frame left correction; applied to original full trajectory',
        'backend_iters':[7,12],'backend_depth_optimized':True,'seed':0,'gt_used_in_inference':False,
        'source':str(out/'ba_inputs.npz'),'args':vars(args),'repeat':rep}
    (out/'provenance.json').write_text(json.dumps(config,indent=2))
    runtime={'feature_seconds':feature_seconds,'backend_seconds':backend_seconds,'interpolation_seconds':interp_seconds,
        'total_droid_seconds':feature_seconds+backend_seconds+interp_seconds,
        'preprocessing_seconds':prep_seconds,'model_load_seconds':load_seconds,
        'keyframes':count,'graphs':graphs,'cuda_ba_calls':ba_calls[0],'stage_times':stages,
        'normalization_factors':normalizations,'normalization_scale':scale[0],'backend_errors':system.backend.errors}
    (out/'runtime.json').write_text(json.dumps(runtime,indent=2))
    (out/'BA_SUCCESS.json').write_text(json.dumps(config,indent=2))
    print('FINISHED',json.dumps(runtime),flush=True)
if __name__=='__main__':main()
