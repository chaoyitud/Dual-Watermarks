from .hf import HFServer
from .server import Server



def get_model(inference_engine, config, **kwargs):
    if inference_engine == "vllm":
        raise NotImplementedError("VLLM is not implemented")
    elif inference_engine == "hf":
        return HFServer(config, **kwargs)
    else:
        raise NotImplementedError(f"{inference_engine} is not implemented")


ALL = ["Server", "VLLMServer", "HFServer", "get_model"]
