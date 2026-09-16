"""Pinned official feature networks with bounded sparse matching and local weights."""
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse
import sys
import numpy as np
from .learned_assets import BACKENDS, verify_assets


def canonical_state(state,layers):
    renamed={}
    for key,value in state.items():
        # Official LighterGlue checkpoint also stores the separately loaded XFeat extractor.
        # Keep strict validation of every matcher parameter.
        if key.startswith("extractor.model."):
            continue
        key=key.removeprefix("matcher.")
        for i in range(layers):
            key=key.replace(f"self_attn.{i}.",f"transformers.{i}.self_attn.")
            key=key.replace(f"cross_attn.{i}.",f"transformers.{i}.cross_attn.")
        renamed[key]=value
    return renamed


class LearnedMatcher:
    def __init__(self,root,config):
        root=Path(root);self.cfg=config;self.name=config["backend"]
        verify_assets(root,self.name)
        # Paths belong to this node process and this isolated workspace.
        for directory in ["deps/learned-python","vendor/XFeat","vendor/LightGlue"]:
            sys.path.insert(0,str(root/directory))
        import torch
        self.torch=torch
        device=config.get("device","cuda")
        if device not in ("cpu","cuda"):raise ValueError("device must be cpu or cuda")
        if device=="cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; choose an explicit CPU profile after checking its timing budget")
        self.device=torch.device(device)
        torch.set_num_threads(min(4,max(1,int(config.get("cpu_threads",2)))))
        self.limit=int(config.get("max_keypoints",1024))
        if not 32<=self.limit<=4096:raise ValueError("Feature budget must be between 32 and 4096")
        self.mixed=bool(config.get("mixed_precision",False)) and device=="cuda"
        self.matcher=None
        if self.name.startswith("xfeat"):
            from modules.xfeat import XFeat
            self.extractor=XFeat(weights=str(root/"vendor/XFeat/weights/xfeat.pt"),
                top_k=self.limit,device=device).eval()
            if self.name=="xfeat_lighterglue":
                from modules.lighterglue import LighterGlue
                from kornia.feature.lightglue import LightGlue
                conf=dict(LighterGlue.default_conf_xfeat)
                conf.update(flash=bool(config.get("flash_attention",True)),
                    mp=self.mixed,filter_threshold=float(config.get("match_threshold",.1)))
                self.matcher=LightGlue(features=None,**conf)
                state=torch.load(root/"vendor/XFeat/weights/xfeat-lighterglue.pt",
                    map_location="cpu",weights_only=True)
                self._strict_matcher_weights(state)
        else:
            from lightglue import LightGlue,SuperPoint,ALIKED
            feature=self.name.split("_")[0]
            with self._local_checkpoints(root/"deps/learned-checkpoints"):
                self.extractor=(SuperPoint(max_num_keypoints=self.limit) if feature=="superpoint"
                    else ALIKED(model_name="aliked-n16",max_num_keypoints=self.limit))
                self.matcher=LightGlue(features=feature,flash=bool(config.get("flash_attention",True)),
                    mp=self.mixed,filter_threshold=float(config.get("match_threshold",.1)),
                    depth_confidence=float(config.get("depth_confidence",.9)),
                    width_confidence=float(config.get("width_confidence",.95)))
            state=torch.load(root/"deps/learned-checkpoints"/(feature+"_lightglue_v0-1_arxiv.pth"),
                map_location="cpu",weights_only=True)
            self._strict_matcher_weights(state)
        self.extractor=self.extractor.to(self.device).eval()
        if self.matcher is not None:self.matcher=self.matcher.to(self.device).eval()
        if self.device.type=="cuda":torch.cuda.reset_peak_memory_stats(self.device)

    def _strict_matcher_weights(self,state):
        state=canonical_state(state,self.matcher.conf.n_layers)
        incompatible=self.matcher.load_state_dict(state,strict=False)
        missing=set(incompatible.missing_keys)-{"confidence_thresholds"}
        if missing or incompatible.unexpected_keys:
            raise RuntimeError("Matcher architecture/weights mismatch: "+
                str((sorted(missing),incompatible.unexpected_keys)))

    @contextmanager
    def _local_checkpoints(self,directory):
        # Upstream constructors use torch.hub; constrain them to preverified local files.
        original=self.torch.hub.load_state_dict_from_url
        def local(url,*args,file_name=None,**kwargs):
            filename=file_name or Path(urlparse(url).path).name
            path=directory/filename
            if not path.is_file():raise RuntimeError("Offline checkpoint missing: "+filename)
            return self.torch.load(path,map_location="cpu",weights_only=True)
        self.torch.hub.load_state_dict_from_url=local
        try:yield
        finally:self.torch.hub.load_state_dict_from_url=original

    def _sync(self):
        if self.device.type=="cuda" and self.cfg.get("measure_stages",True):
            self.torch.cuda.synchronize(self.device)

    def extract(self,gray):
        torch=self.torch
        if gray.ndim!=2 or gray.dtype!=np.uint8:raise ValueError("Expected mono8 input")
        tensor=torch.from_numpy(np.ascontiguousarray(gray)).to(self.device,dtype=torch.float32)[None,None]/255.
        if self.name=="aliked_lightglue":tensor=tensor.expand(-1,3,-1,-1)
        with torch.inference_mode(),torch.autocast(self.device.type,enabled=self.mixed):
            if self.name.startswith("xfeat"):
                raw=self.extractor.detectAndCompute(tensor,top_k=self.limit)[0]
                data={key:raw[key][None] for key in ("keypoints","descriptors")}
                data["image_size"]=torch.tensor([[gray.shape[1],gray.shape[0]]],
                    device=self.device,dtype=torch.float32)
            else:
                raw=self.extractor.extract(tensor,resize=None)
                data={key:raw[key] for key in ("keypoints","descriptors","image_size")}
            for key in ("keypoints","descriptors"):data[key]=data[key][:,:self.limit].contiguous()
        self._sync()
        if not torch.isfinite(data["descriptors"]).all():raise ValueError("Nonfinite learned descriptors")
        pixels=data["keypoints"][0].float().cpu().numpy()
        if not np.isfinite(pixels).all():raise ValueError("Nonfinite learned keypoints")
        return dict(data=data,pixels=pixels)

    def match(self,first,second):
        if min(len(first["pixels"]),len(second["pixels"]))<2:
            return np.empty((0,2),dtype=np.int64)
        torch=self.torch
        with torch.inference_mode(),torch.autocast(self.device.type,enabled=self.mixed):
            if self.name=="xfeat_mnn":
                a=first["data"]["descriptors"][0];b=second["data"]["descriptors"][0]
                similarity=(a@b.T).clamp(-1.,1.)
                best,indices=similarity.topk(2,dim=1)
                reverse=similarity.argmax(dim=0)
                rows=torch.arange(len(a),device=self.device)
                keep=(reverse[indices[:,0]]==rows)&(best[:,0]>=self.cfg.get("min_cosine_similarity",.82))
                ratio=float(self.cfg.get("nearest_neighbor_ratio",.8))
                keep &= (1-best[:,0]) <= ratio**2*(1-best[:,1]).clamp_min(1e-6)
                matches=torch.stack((rows[keep],indices[keep,0]),dim=1)
            else:
                result=self.matcher({"image0":first["data"],"image1":second["data"]})
                matches=result["matches"][0]
        self._sync()
        return matches.detach().cpu().numpy().astype(np.int64,copy=False).reshape(-1,2)

    def resource_snapshot(self):
        models=[self.extractor]+([self.matcher] if self.matcher is not None else [])
        result=dict(backend=self.name,device=str(self.device),max_keypoints=self.limit,
            torch_version=str(self.torch.__version__),torch_cuda_build=self.torch.version.cuda,
            model_parameter_bytes=sum(p.numel()*p.element_size() for m in models for p in m.parameters()))
        if self.device.type=="cuda":
            result.update(cuda_allocated_bytes=self.torch.cuda.memory_allocated(self.device),
                cuda_reserved_bytes=self.torch.cuda.memory_reserved(self.device),
                cuda_peak_allocated_bytes=self.torch.cuda.max_memory_allocated(self.device))
        return result
