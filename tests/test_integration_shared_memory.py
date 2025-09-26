import time

from wave_monitor.client import WaveMonitor
from wave_monitor.window import MonitorWindow, SharedServerState


def test_server_client_shared_memory_integration(qapp):
    monitor = MonitorWindow()
    client = WaveMonitor(create_window=False)
    try:
        # Initial default
        assert abs(client.get_wfm_interval() - SharedServerState.DEFAULT_WFM_INTERVAL) < 1e-6

        for val in (0.5, 1.2, 0.8):
            monitor.state.wfm_interval = val
            # allow a tiny delay for client to read new value (though direct read)
            time.sleep(0.05)
            assert abs(client.get_wfm_interval() - val) < 1e-6
    finally:
        client.close()
        monitor.window.close()