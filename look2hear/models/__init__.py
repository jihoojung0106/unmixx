from .tiger import TIGER
#from .tiger10 import TIGER10
from .base_model import BaseModel
#from .djcm_module 
#from .djcm_module import DJCMPE_Decoder, DJCMEncoder, DJCMLatentBlocks

# from .discriminator import DISCRIMINATOR
# from .metricgan import MetricDiscriminator
# from .tiger_phase1 import TIGERPHASE1
from .tiger_phase2 import TIGERPHASE2
# from .tiger_phase3 import TIGERPHASE3
# from .tiger_gan import TIGERGAN1
# from .tigermoe2 import TIGERMOE2
# from .tiger_att1 import TIGERATT1
# from .tiger_att2 import TIGERATT2
from .tiger_att3 import TIGERATT3
from .tiger_att4 import TIGERATT4
from .tiger_all import TIGERALL
from .tiger_all2 import TIGERALL2
__all__ = [
    "TIGER"
]

def register_model(custom_model):
    """Register a custom model, gettable with `models.get`.

    Args:
        custom_model: Custom model to register.

    """
    if (
        custom_model.__name__ in globals().keys()
        or custom_model.__name__.lower() in globals().keys()
    ):
        raise ValueError(
            f"Model {custom_model.__name__} already exists. Choose another name."
        )
    globals().update({custom_model.__name__: custom_model})


def get(identifier):
    """Returns an model class from a string (case-insensitive).

    Args:
        identifier (str): the model name.

    Returns:
        :class:`torch.nn.Module`
    """
    if isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret model name : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret model name : {str(identifier)}")
