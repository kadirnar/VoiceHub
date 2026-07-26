from enum import Enum

class Backend(Enum):
    HF = "hf"
    LLAMACPP = "llamacpp"
    EXL2 = "exl2"
    EXL2ASYNC = "exl2async"
    VLLM = "vllm"
    LLAMACPP_SERVER = "llamacpp_server"
    LLAMACPP_ASYNC_SERVER = "llamacpp_async_server"

class InterfaceVersion(Enum):
    V1 = 1
    V2 = 2
    V3 = 3

class Models(Enum):
    VERSION_0_1_SIZE_350M = "OuteTTS-0.1-350M"
    VERSION_0_2_SIZE_500M = "OuteTTS-0.2-500M"
    VERSION_0_3_SIZE_500M = "OuteTTS-0.3-500M"
    VERSION_0_3_SIZE_1B = "OuteTTS-0.3-1B"
    VERSION_1_0_SIZE_1B = "Llama-OuteTTS-1.0-1B"
    VERSION_1_0_SIZE_0_6B = "OuteTTS-1.0-0.6B"

class GenerationType(Enum):
    REGULAR = "regular"
    CHUNKED = "chunked"
    GUIDED_WORDS = "guided_words"
    STREAM = "stream"
    BATCH = "batch"

class LlamaCppQuantization(Enum):
    FP16 = "FP16"
    Q8_0 = "Q8_0"
    Q6_K = "Q6_K"
    Q5_K_S = "Q5_K_S"
    Q5_K_M = "Q5_K_M"
    Q5_1 = "Q5_1"
    Q5_0 = "Q5_0"
    Q4_K_S = "Q4_K_S"
    Q4_K_M = "Q4_K_M"
    Q4_1 = "Q4_1"
    Q4_0 = "Q4_0"
    Q3_K_S = "Q3_K_S"
    Q3_K_M = "Q3_K_M"
    Q3_K_L = "Q3_K_L"
    Q2_K = "Q2_K"

MODEL_INFO = {
    Models.VERSION_0_1_SIZE_350M: {
        "max_seq_length": 4096,
        "interface_version": InterfaceVersion.V1, 
    },
    Models.VERSION_0_2_SIZE_500M: {
        "max_seq_length": 4096,
        "interface_version": InterfaceVersion.V2,
    },
    Models.VERSION_0_3_SIZE_500M: {
        "max_seq_length": 4096,
        "interface_version": InterfaceVersion.V2,
    },
    Models.VERSION_0_3_SIZE_1B: {
        "max_seq_length": 4096,
        "interface_version": InterfaceVersion.V2,
    },
    Models.VERSION_1_0_SIZE_1B: {
        "max_seq_length": 8192,
        "interface_version": InterfaceVersion.V3,
    },
    Models.VERSION_1_0_SIZE_0_6B: {
        "max_seq_length": 8192,
        "interface_version": InterfaceVersion.V3,
    },
}