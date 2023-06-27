import pytest
from rfi_scan_analysis import csv_utils as ut

#Gaussian object for testing integration and fitting
@pytest.fixture
def clean_gauss_test_data():
    return ut.gen_gauss_test_data(-10, 10, 0, 1, 1, 10000, 0)

#Noisy gaussian object for testing integration and fitting
#@pytest.fixture
def noisy_gauss_test_data():
    return ut.gen_gauss_test_data(-10, 10, 0, 1, 1, 10000, .1)

#Test 68-95-99.7 rule when itegrating under a standard normal distribution
def test_std_norm_integrate(clean_gauss_test_data):
    std1 = ut.integrate_range(ut.trim_data(clean_gauss_test_data, -1, 1, "test_data"))
    assert std1 == .68
    std2 = ut.integrate_range(ut.trim_data(clean_gauss_test_data, -2, 2, "test_data"))
    assert std2 == .95
    std3 = ut.integrate_range(ut.trim_data(clean_gauss_test_data, -3, 3, "test_data"))
    assert std3 == .997


