from loguru import logger
import os

from .version.interface import (
    InterfaceHF, InterfaceLLAMACPP, 
    InterfaceEXL2, InterfaceEXL2Async,
    InterfaceVLLMBatch, 
    InterfaceLlamaCPPServer, InterfaceLlamaCPPServerAsyncBatch
)
from .models.config import ModelConfig
from .models.info import Backend, InterfaceVersion

def Interface(config: ModelConfig) -> (
        InterfaceHF | InterfaceLLAMACPP | 
        InterfaceEXL2 | InterfaceEXL2Async |
        InterfaceVLLMBatch |
        InterfaceLlamaCPPServer | InterfaceLlamaCPPServerAsyncBatch):

    if config.backend == Backend.HF:
        return InterfaceHF(config)
    elif config.backend == Backend.LLAMACPP:
        return InterfaceLLAMACPP(config)
    elif config.backend == Backend.EXL2:
        return InterfaceEXL2(config)
    elif config.backend == Backend.EXL2ASYNC:
        return InterfaceEXL2Async(config)
    elif config.backend == Backend.VLLM:
        warning_msg = "VLLM backend is experimental and may cause issues with audio generation."
        logger.warning(warning_msg)
        return InterfaceVLLMBatch(config)
    elif config.backend == Backend.LLAMACPP_SERVER:
        return InterfaceLlamaCPPServer(config)
    elif config.backend == Backend.LLAMACPP_ASYNC_SERVER:
        return InterfaceLlamaCPPServerAsyncBatch(config)

    raise ValueError(f"Invalid backend: {config.backend} - must be one of {list(Backend)}")
