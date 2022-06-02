import numpy as np
from scipy.io import wavfile
from util import *
import pytest

@pytest.mark.parametrize("test_input", [-1, 0, 1])
def test_mulaw(test_input):
    n = np.array([test_input])
    assert mulaw(n) == n

@pytest.mark.parametrize("test_input", [-1, 0, 1])
def test_inv_mulaw(test_input):
    n = np.array([test_input])
    assert inv_mulaw(n) == n

@pytest.mark.parametrize("test_input", [-1, -0.5, -0.25, 0, 0.25, 0.5, 1])
def test_mulaw_inv_mulaw(test_input):
    n = np.array([test_input])
    np.testing.assert_allclose(inv_mulaw(mulaw(n)), n)

@pytest.mark.parametrize("test_input, expected", [[-1, 0], [0, 127], [1, 255]])
def test_mulaw_quantize(test_input, expected):
    test_input_array = np.array([test_input])
    expected_array = np.array([expected])
    assert mulaw_quantize(test_input_array) == expected_array

@pytest.mark.parametrize("test_input, expected", [[0, -1], [127, 0], [255, 1]])
def test_inv_mulaw_quantize(test_input, expected):
    test_input_array = np.array([test_input])
    expected_array = np.array([expected])
    np.testing.assert_allclose(inv_mulaw_quantize(test_input_array), expected_array, atol=1e-4)

@pytest.mark.parametrize("test_input", [-1, -0.5, -0.25, 0, 0.25, 0.5, 1])
def test_inv_mulaw_quantize(test_input):
    test_input_array = np.array([test_input])
    result_array = inv_mulaw_quantize(mulaw_quantize(test_input_array))
    np.testing.assert_allclose(result_array, test_input_array, atol=2e-2)