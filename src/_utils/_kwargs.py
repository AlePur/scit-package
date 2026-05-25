import numpy as np

def _format_kwargs(user_provided: dict, default: dict) -> dict:
    allkeys = np.concat((list(default.keys()), list(user_provided.keys())))
    allkeys = np.unique(allkeys)

    for k in allkeys:
        if (k in user_provided.keys()):
            default[k] = user_provided[k]
    return default
    