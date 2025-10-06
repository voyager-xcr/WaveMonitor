import multiprocessing.shared_memory as shared_memory

from wave_monitor.constants import SHARED_MEMORY_NAME
from wave_monitor.window import MonitorWindow


def test_wfm_interval_property(qapp):
    mw = MonitorWindow()
    try:
        # Read default value via state
        default_val = mw.state.wfm_interval
        assert isinstance(default_val, float)

        # Update via state property
        mw.state.wfm_interval = 0.7
        assert abs(mw.state.wfm_interval - 0.7) < 1e-6

        # Read underlying shared memory directly
        shm = shared_memory.ShareableList(name=SHARED_MEMORY_NAME)
        try:
            assert abs(float(shm[0]) - 0.7) < 1e-6
        finally:
            shm.shm.close()
    finally:
        mw.window.close()


def test_wfm_interval_update_reflected_in_shared_memory(qapp):
    """Setting state.wfm_interval should reflect in shared memory list."""
    mw = MonitorWindow()
    try:
        mw.state.wfm_interval = 1.1
        shm = shared_memory.ShareableList(name=SHARED_MEMORY_NAME)
        try:
            assert abs(float(shm[0]) - 1.1) < 1e-6
        finally:
            shm.shm.close()
    finally:
        mw.window.close()
