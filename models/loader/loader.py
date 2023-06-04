import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import torch
import transformers
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer, LlamaTokenizer)

from configs.model_config import LLM_DEVICE


class LoaderCheckPonit:
    """
    A class to load model from checkpoint
    """
    
    no_remote_model: bool = False # whether load model from remote
    model_name: str = None
    model_path: str = None
    model_config: str = None
    model_dir: str = None
    model: object = None
    tokenizer: object = None
    lora_names: set = []
    lora_dir: str = None
    ptuning_v2_dir: str = None
    use_ptuning_v2: bool = False
    load_in_8bit: bool = False
    is_llamacpp: bool = False
    bf16: bool = False
    params: dict = None
    device_map: Optional[Dict[str, int]] = None
    llm_device: str = LLM_DEVICE

    def __init__(self, params: dict = None):
        """initialize params"""
        self.model_path = None
        self.model = None
        self.tokenizer = None
        self.params = params or {}
        self.no_remote_model = params.get('no_remote_model', False)
        self.model_name = params.get('model', '')
        self.lora = params.get('lora', '')
        self.use_ptuning_v2 = params.get('use_ptuning_v2', False)
        self.model_dir = params.get('model_dir', '')
        self.lora_dir = params.get('lora_dir', '')
        self.ptuning_dir = params.get('ptuning_dir', 'ptuning-v2')
        self.load_in_8bit = params.get('load_in_8bit', False)
        self.bf16 = params.get('bf16', False)
    
    def _load_config(self, model_name: str) -> object:
        checkpoint = Path(f'{self.model_dir}/{model_name}')
        if self.model_path:
            checkpoint = Path(f'{self.model_path}')
        else:
            if not self.no_remote_model:
                checkpoint = model_name
        model_config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
        return model_config
    
    def _load_model(self, model_name: str) -> object:
        """
        support model:
            chatglm
            moss
            llamacpp
            vicuna
        
        """
        print(f"Loading {model_name} ...")
        t0 = time.time()
        checkpoint = Path(f'{self.model_dir}/{model_name}')
        self.is_llamacpp = len(list(checkpoint.glob('ggml*.bin'))) > 0
        if self.model_path:
            checkpoint = Path(f'{self.model_path}')
        else:
            if not self.no_remote_model:
                checkpoint = model_name
        if "chatglm" in model_name.lower():
            LoaderClass = AutoModel
        else:
            LoaderClass = AutoModelForCausalLM
        # load model
        if not any([self.is_llamacpp, self.load_in_8bit, self.llm_device.lower() == 'cpu']):
            if torch.cuda.is_available() and self.llm_device.lower().startswith("cuda"):
                num_gpus = torch.cuda.device_count()
                if num_gpus < 2 and self.device_map is None:
                    model = LoaderClass.from_pretrained(checkpoint, config=self.config, torch_dtype=torch.bfloat16 if self.bf16 else torch.float16, trust_remote_code=True).half().cuda()
                else:
                    from accelerate  import dispatch_model
                    model = LoaderClass.from_pretrained(checkpoint, config=self.config, torch_dtype=torch.bfloat16 if self.bf16 else torch.float16, trust_remote_code=True).half()
                    if self.device_map is None:
                        if "chatglm" in model_name.lower():
                            self.device_map = self.__chatglm_device_mapper(num_gpus)
                        elif "moss" in model_name.lower():
                            self.device_map = self.__moss_device_mapper(num_gpus)
                        elif "vicuna-13b" in model_name.lower():
                            self.device_map = self.__vicuna_device_mapper(num_gpus)
                    model = dispatch_model(model, device_map=self.device_map)
            else:
                model = LoaderClass.from_pretrained(checkpoint, config=self.config, trust_remote_code=True).float().to(self.llm_device)
        elif self.is_llamacpp:
            pass
        elif self.load_in_8bit:
            pass
        else:
            print("Warning: self.llm_device is False.\nThis means that no use GPU  bring to be load CPU mode\n")
            params = {"low_cpu_mem_usage": True, "torch_dtype": torch.float32, "trust_remote_code": True}
            model = LoaderClass.from_pretrained(checkpoint, **params).to(self.llm_device, dtype=float)
        # load tokenizer
        if type(model) is transformers.LlamaTokenizer:
            tokenizer = LlamaTokenizer.from_pretrained(checkpoint, clean_up_tokenization_spaces=True)
            # TODO
            # Leaving this here until the LLaMA tokenizer gets figured out.
            # For some people this fixes things, for others it causes an error.
            try:
                tokenizer.eos_token_id = 2
                tokenizer.bos_token_id = 1
                tokenizer.pad_token_id = 0
            except Exception as e:
                print(e)
                pass
        else:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        
        print(f"Loaded {model_name} in {time.time() - t0:.2f}s")
        return model, tokenizer
    
    def _add_lora_to_model(self, lora_names: set):
        """merge lora weights with raw weights"""
        pass

    def clear_torch_cache(self):
        pass

    def unload_model(self):
        del self.model
        del self.tokenizer
        self.model, self.tokenizer = None, None
        self.clear_torch_cache()
    
    def set_model_path(self, model_path: str) -> None:
        self.model_path = model_path
    
    def reload_model(self):
        pass

    def __chatglm_device_mapper(self, num_gpus: int) -> Dict[str, int]:
        num_trans_layers = 28
        per_gpu_layers = 30 / num_gpus
        # bugfix: PEFT加载lora模型出现的层命名不同
        if self.lora:
            layer_prefix = 'base_model.model.transformer'
        else:
            layer_prefix = 'transformer'
        # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
        # windows下 model.device 会被设置成 transformer.word_embeddings.device
        # linux下 model.device 会被设置成 lm_head.device
        # 在调用chat或者stream_chat时,input_ids会被放到model.device上
        # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
        # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
        device_map = {f'{layer_prefix}.word_embeddings': 0,
                      f'{layer_prefix}.final_layernorm': 0, 'lm_head': 0,
                      f'base_model.model.lm_head': 0, }
        used = 2
        gpu_target = 0
        for i in range(num_trans_layers):
            if used >= per_gpu_layers:
                gpu_target += 1
                used = 0
            assert gpu_target < num_gpus
            device_map[f'{layer_prefix}.layers.{i}'] = gpu_target
            used += 1
        return device_map
    
    def __moss_device_mapper(self, num_gpus: int) -> Dict[str, int]:
        try:
            from accelerate import init_empty_weights
            from accelerate.utils import get_balanced_memory, infer_auto_device_map
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            from transformers.modeling_utils import no_init_weights
            from transformers.utils import ContextManagers
        except ImportError as exc:
            raise ValueError(
                "Could not import depend python package "
                "Please install it with `pip install transformers` "
                "`pip install bitsandbytes``pip install accelerate`."
            ) from exc
        checkpoint = Path(f'{self.model_dir}/{model_name}')
        if self.model_path:
            checkpoint = Path(f'{self.model_path}')
        else:
            if not self.no_remote_model:
                checkpoint = model_name
        cls = get_class_from_dynamic_module(class_reference="fnlp/moss-moon-003-sft--modeling_moss.MossForCausalLM",
                                            pretrained_model_name_or_path=checkpoint)
        with ContextManagers([no_init_weights(_enable=True), init_empty_weights()]):
            model = cls(self.model_config)
            max_memory = get_balanced_memory(model, dtype=torch.int8 if self.load_in_8bit else None,
                                             low_zero=False, no_split_module_classes=model._no_split_modules)
            device_map = infer_auto_device_map(
                model, dtype=torch.float16 if not self.load_in_8bit else torch.int8, max_memory=max_memory,
                no_split_module_classes=model._no_split_modules)
            device_map["transformer.wte"] = 0
            device_map["transformer.drop"] = 0
            device_map["transformer.ln_f"] = 0
            device_map["lm_head"] = 0
            return device_map
    
    def __vicuna_device_mapper(self, num_gpus: int) -> Dict[str, int]:
        pass
