from typing import Union
import nvdiffrast.torch as dr

RasterizeContext = Union[dr.RasterizeCudaContext, dr.RasterizeGLContext]
