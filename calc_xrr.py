# use provided fit parameters to calculate xrr-cuves
from refl1d.reflectivity import reflectivity as refl1d_engine
from tqdm import tqdm
import numpy as np


def calc_reflectivity(thicknesses, roughnesses, slds, q_values, progress_bar=False):
    """ 
        code slipplet taken from mlreflect package
        """
    number_of_q_values = q_values.shape[1]
    number_of_curves = thicknesses.shape[0]

    reflectivity_curves = np.zeros([number_of_curves, number_of_q_values])
    depth = np.fliplr(thicknesses)
    depth = np.hstack(
        (np.ones((number_of_curves, 1)), depth, np.ones((number_of_curves, 1)))
    )
    rho = np.fliplr(slds)

    for curve in tqdm(range(number_of_curves), disable=not progress_bar):
        params = {
            "kz": q_values[curve, :] / 2,
            "depth": depth[curve, :],
            "sigma": np.flip(roughnesses[curve, :]),
        }

        this_rho = rho[curve, :]
        if np.sum(np.iscomplex(this_rho)) > 0:
            irho = this_rho.imag
            this_rho = this_rho.real
            params["irho"] = irho
        params["rho"] = this_rho

        reflectivity = refl1d_engine(**params)
        del params
        reflectivity_curves[curve, :] = reflectivity
    return reflectivity_curves

def _fill(n,data):
    """
    make sure that 1d data is returned. If input data is 0d it will be replicated n times.
    """
    
    if data.ndim == 0:
        return  np.full(n, data)
    else:
        return data

def prep_model(q, dataset):
    """
    prepare a tabular view for a stack of fit parameters.
    """
    n = dataset["fit"]["Film_thickness"].shape[0]
    thicknesses = np.array(
        [_fill(n, dataset["fit"]["SiOx_thickness"]), dataset["fit"]["Film_thickness"]]
    ).transpose()
    roughnesses = np.array(
        [
            _fill(n, dataset["fit"]["Si_roughness"]),
            _fill(n, dataset["fit"]["SiOx_roughness"]),
            dataset["fit"]["Film_roughness"],
        ]
    ).transpose()
    slds = np.array(
        [
            _fill(n, dataset["fit"]["Si_sld"]),
            _fill(n, dataset["fit"]["SiOx_sld"]),
            dataset["fit"]["Film_sld"],
            np.zeros(n),
        ]
    ).transpose()
    q_values = np.array([q] * n)
    
    return thicknesses, roughnesses, slds, q_values
