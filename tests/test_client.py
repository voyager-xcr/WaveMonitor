import warnings

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


def test_get_wfm_interval_from_shared_memory(monkeypatch):
    """Test that get_wfm_interval reads from shared memory"""
    import multiprocessing.shared_memory as shared_memory

    from wave_monitor.constants import SHARED_MEMORY_NAME
    
    # Create a test shared memory
    test_interval = 2.5
    wm = None
    shm = None
    try:
        # Clean up any existing shared memory
        try:
            old_shm = shared_memory.ShareableList(name=SHARED_MEMORY_NAME)
            old_shm.shm.close()
            old_shm.shm.unlink()
        except FileNotFoundError:
            pass

        # Create new shared memory with test value
        shm = shared_memory.ShareableList([test_interval], name=SHARED_MEMORY_NAME)

        # Test client reading from shared memory
        wm = WaveMonitor(create_window=False)
        res = wm.get_wfm_interval()

        assert isinstance(res, float)
        assert abs(res - test_interval) < 1e-6

        # Test fallback when shared memory is not available
        wm._shared_memory = None
        shm.shm.close()
        shm.shm.unlink()

        # Should return default fallback value (client fallback is 0.0 when shared memory not available)
        res = wm.get_wfm_interval()
        assert isinstance(res, float)
        assert res == 0.0
    finally:
        if wm is not None:
            wm.close()
        if shm is not None:
            try:
                shm.shm.close()
                shm.shm.unlink()
            except Exception:
                pass
