from dataset_specifications.normal import NormalSet
from dataset_specifications.double_normal import DoubleNormalSet
from dataset_specifications.linear import Linear
from dataset_specifications.sinus import Sinus
from dataset_specifications.const_noise import ConstNoiseSet
from dataset_specifications.syn_poisson import SynPoissonSet
from dataset_specifications.syn_tweedie import SynTweedieSet
from dataset_specifications.syn_normal import SynNormalSet
from dataset_specifications.syn_logistic import SynLogisticSet
from dataset_specifications.heteroskedastic import HeteroskedasticSet
from dataset_specifications.bimodal import BimodalSet
from dataset_specifications.exponential import ExponentialSet
from dataset_specifications.laplace import LaplaceSet
from dataset_specifications.microwave import MicroWaveSet
from dataset_specifications.wine import WineSet
from dataset_specifications.complex import ComplexSet
from dataset_specifications.mixture_2d import Mixture2DSet
from dataset_specifications.swirls import SwirlsSet
from dataset_specifications.power import PowerSet
from dataset_specifications.butterfly import ButterflySet
from dataset_specifications.housing import HousingSet
from dataset_specifications.diabetes import DiabetesSet
from dataset_specifications.house_age import HouseAgeSet
from dataset_specifications.trajectories import TRAJECTORIES_SET_DICT
from dataset_specifications.wmix import WMIX_SET_DICT

from dataset_specifications.real_insurance import RealInsuranceSet

# List of all available datasets
sets = {
    "normal": NormalSet,
    "double_normal": DoubleNormalSet,
    "linear": Linear,
    "sinus": Sinus,
    "const_noise": ConstNoiseSet,
    "syn_poisson": SynPoissonSet,
    "syn_tweedie": SynTweedieSet,
    "syn_normal": SynNormalSet,
    "syn_logistic": SynLogisticSet,
    "heteroskedastic": HeteroskedasticSet,
    "bimodal": BimodalSet,
    "exponential": ExponentialSet,
    "laplace": LaplaceSet,
    "microwave": MicroWaveSet,
    "wine": WineSet,
    "complex": ComplexSet,
    "mixture_2d": Mixture2DSet,
    "swirls": SwirlsSet,
    "power": PowerSet,
    "butterfly": ButterflySet,
    "housing": HousingSet,
    "diabetes": DiabetesSet,
    "house_age": HouseAgeSet,

    "real_insurance": RealInsuranceSet,
}

# There exists multiple trajectories and wmix datasets,
# include them all from original file
sets.update(TRAJECTORIES_SET_DICT)
sets.update(WMIX_SET_DICT)


def get_dataset_spec(name):
    return sets[name]

