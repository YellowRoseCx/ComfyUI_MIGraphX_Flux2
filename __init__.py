from .migraphx_compile_sd3 import CompileDiffusersMIGraphX
from .migraphx_compile_flux2 import CompileFlux2MIGraphX
from .migraphx_utils import MgxTransformer

NODE_CLASS_MAPPINGS = {
    "CompileDiffusersMIGraphX": CompileDiffusersMIGraphX,
    "CompileFlux2MIGraphX": CompileFlux2MIGraphX
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CompileDiffusersMIGraphX": "Compile diffusion model on migraphx",
    "CompileFlux2MIGraphX": "Compile Flux2 model on migraphx"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']