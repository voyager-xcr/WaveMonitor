import warnings

import msgpack
import msgpack_numpy
import numpy as np
import pytest

from wave_monitor.client import WaveMonitor


def test_add_wfm_type_and_shape_checks():
    wm = WaveMonitor(create_window=False)

    # name must be str
    with pytest.raises(TypeError):
        wm.add_wfm(123, np.array([0, 1]), [np.array([0, 1])])

    # t must be numpy array
    with pytest.raises(TypeError):
        wm.add_wfm("n", [0, 1], [np.array([0, 1])])

    # ys must be list
    with pytest.raises(TypeError):
        wm.add_wfm("n", np.array([0, 1]), np.array([0, 1]))

    # t must be 1D
    with pytest.raises(ValueError):
        wm.add_wfm("n", np.zeros((2, 2)), [np.zeros((2,))])

    # ys elements must be numpy arrays
    with pytest.raises(TypeError):
        wm.add_wfm("n", np.array([0, 1]), ["bad"])

    # ys elements must be 1D
    with pytest.raises(ValueError):
        wm.add_wfm("n", np.array([0, 1]), [np.zeros((1, 1))])

    # ys elements must match shape of t
    with pytest.raises(ValueError):
        wm.add_wfm("n", np.array([0, 1, 2]), [np.array([0, 1])])


def test_add_line_delegates_to_add_wfm(monkeypatch):
    wm = WaveMonitor(create_window=False)

    called = {}

    def fake_add_wfm(name, t, ys):
        called['args'] = (name, t.copy(), [y.copy() for y in ys])

    monkeypatch.setattr(wm, "add_wfm", fake_add_wfm)

    t = np.array([0, 1])
    ys = [np.array([0, 1])]
    # add_line is marked deprecated; suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wm.add_line("n", t, ys, offset=0)

    assert 'args' in called
    assert called['args'][0] == "n"
    np.testing.assert_array_equal(called['args'][1], t)
    np.testing.assert_array_equal(called['args'][2][0], ys[0])


def test_get_wfm_interval_parses_reply(monkeypatch):
    wm = WaveMonitor(create_window=False)

    payload = {"_type": "wfm_interval", "interval": 1.23}
    packed = msgpack.packb(payload, default=msgpack_numpy.encode)

    def fake_query(msg, timeout_ms=200):
        return packed

    monkeypatch.setattr(wm, "query", fake_query)
    res = wm.get_wfm_interval()
    assert isinstance(res, float)
    assert abs(res - 1.23) < 1e-6
