from numpy import std, mean, sqrt
from statsmodels.stats.power import TTestIndPower
from scipy import stats


def calculate_ttest_sample_size(effect_size, alpha=0.05, power=0.80):
    # Create an instance of the FTestAnovaPower class
    power_analysis = TTestIndPower()

    # Calculate sample size
    sample_size = power_analysis.solve_power(effect_size=effect_size,
                                             alpha=alpha,
                                             power=power,
                                             alternative="two-sided",
                                             ratio=1,  # ratio of sample sizes between the two groups
                                             nobs1=None,  # sample size for group 1, None means it will be calculated
                                             )
    return sample_size


def calculate_ttest_effect_size(a, b):
    """
    cohen d is the effect size for t-test
    :param a:
    :param b:
    :return:
    """
    nx = len(a)
    ny = len(b)
    dof = nx + ny - 2

    effect_size = (mean(a) - mean(b)) / sqrt(((nx - 1) * std(a, ddof=1) ** 2 + (ny - 1) * std(b, ddof=1) ** 2) / dof)

    effect_size = abs(effect_size)
    return effect_size

def perform_ttest(a, b):
    t, p = stats.ttest_ind(a, b, alternative="two-sided")
    effect_size = calculate_ttest_effect_size(a, b)

    return t, p, effect_size