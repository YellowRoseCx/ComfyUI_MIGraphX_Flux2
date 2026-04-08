import comfy.model_patcher
import comfy.model_management
import comfy.supported_models
import comfy.model_base
from .migraphx_utils import load_MGX_transformer_model

class CompileFlux2MIGraphX:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model": ("MODEL", ),
                "force_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When set on false, it will try to load model from mxr file, if file exists.",},),
                "model_type": (["flux2"], ),
                "max_batch_size": ("INT", {
                    "default": 4, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Maximum batch size for dynamic shape compilation.",
                },),
                "max_width": ("INT", {
                    "default": 2048, "min": 128, "max": 8192, "step": 128,
                    "tooltip": "Maximum width for dynamic shape compilation.",
                },),
                "max_height": ("INT", {
                    "default": 2048, "min": 128, "max": 8192, "step": 128,
                    "tooltip": "Maximum height for dynamic shape compilation.",
                },),
                "data_type": (["fp16", "fp32"], {
                    "tooltip": "GFX1030 does not natively support bfp16. Use fp16 or fp32.",
                },),
            },
        }
        return inputs

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_NODE = True
    FUNCTION = "compile_on_MIGraphX"
    CATEGORY = "advanced/migraphx"

    def compile_on_MIGraphX(self, model, force_compile, model_type, max_batch_size, max_height, max_width, data_type):
        if model_type == "flux2":
            # Real flux config requires hidden_size, etc., so we fetch it from the original model.
            conf = model.model.model_config
            conf.unet_config["disable_unet_model_creation"] = True
            comfy_model = model.model.__class__(conf)
        else:
            print("ERROR: model not supported.")
            return ()

        mxr_file_name = f"{model_type}_{max_batch_size}_{max_width}_{max_height}_{data_type}.mxr"

        comfy_model.diffusion_model = load_MGX_transformer_model(model, force_compile, mxr_file_name,
                                                                max_batch_size, max_height, max_width, False,
                                                                data_type)
        comfy_model.memory_required = lambda *args, **kwargs: 0

        return (comfy.model_patcher.ModelPatcher(comfy_model,
                                                load_device=comfy.model_management.get_torch_device(),
                                                offload_device=comfy.model_management.unet_offload_device()),)

NODE_CLASS_MAPPINGS = {
    "CompileFlux2MIGraphX": CompileFlux2MIGraphX
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CompileFlux2MIGraphX": "Compile Flux2 model on migraphx"
}
