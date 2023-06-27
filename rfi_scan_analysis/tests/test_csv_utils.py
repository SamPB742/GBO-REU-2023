import pytest
from rfi_scan_analysis import csv_utils as ut

#standard Gaussian data for testing integration and fitting
@pytest.fixture
def clean_gauss_test_data():
    return ut.gen_gauss_test_data(-10, 10, 0, 1, 1, 10000, 0)

#Noisy standard gaussian data for testing integration and fitting
@pytest.fixture
def noisy_gauss_test_data():
    return ut.gen_gauss_test_data(-10, 10, 0, 1, 1, 10000, .1)

#Test 68-95-99.7 rule when itegrating under a standard normal distribution
def test_std_norm__clean_integrate(clean_gauss_test_data):
    std1 = ut.integrate_range(ut.trim_data(clean_gauss_test_data, -1, 1, "test_data"))
    assert abs(std1 - .68) < .01
    std2 = ut.integrate_range(ut.trim_data(clean_gauss_test_data, -2, 2, "test_data"))
    assert abs(std2 - .95) < .01
    std3 = ut.integrate_range(ut.trim_data(clean_gauss_test_data, -3, 3, "test_data"))
    assert abs(std3 - .997) < .01

#Test 68-95-99.7 rule when itegrating under a standard normal distribution
def test_std_norm_noisy_integrate(noisy_gauss_test_data):
    std1 = ut.integrate_range(ut.trim_data(noisy_gauss_test_data, -1, 1, "test_data"))
    assert abs(std1 - .68) < .05
    std2 = ut.integrate_range(ut.trim_data(noisy_gauss_test_data, -2, 2, "test_data"))
    assert abs(std2 - .95) < .05
    std3 = ut.integrate_range(ut.trim_data(noisy_gauss_test_data, -3, 3, "test_data"))
    assert abs(std3 - .997) < .05

#test fit on a clean gaussian shape
def test_simple_gauss_clean_fit(clean_gauss_test_data):
    mean, std, scale = ut.cont_gauss_fit(ut.trim_data(clean_gauss_test_data, -3, 3, "test_data"))
    assert abs(mean) < .01 and abs(1 - std) < .01 and abs(1 - scale) < .01

#test fit on a clean gaussian shape
def test_simple_gauss_noisy_fit(noisy_gauss_test_data):
    mean, std, scale = ut.cont_gauss_fit(ut.trim_data(noisy_gauss_test_data, -3, 3, "test_data"))
    assert abs(mean) < .05 and abs(1 - std) < .05 and abs(1 - scale) < .05
