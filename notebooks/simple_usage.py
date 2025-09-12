import logging

import numpy as np

from wave_monitor import WaveMonitor

logging.basicConfig(level=logging.DEBUG)

monitor = WaveMonitor()
# monitor.clear()

t = np.linspace(0, 1, 1_001)  # 1m pts ~= 1ms for 1GSa/s.
n = 20
i_waves = [np.cos(2 * np.pi * f * t) for f in range(1, n + 1)]
q_waves = [np.sin(2 * np.pi * f * t) for f in range(1, n + 1)]

for i, (i_wave, q_wave) in enumerate(zip(i_waves, q_waves)):
    monitor.add_wfm(
        f"wave_{i}",
        [0, 0, 1050, 1050, 1050, 1050],
        [i_wave, q_wave, i_wave - 1.5, q_wave - 1.5, i_wave + 1.5, q_wave + 1.5],
    )
monitor.autoscale()

monitor.add_wfm(
    "wave_1", t, [i_waves[-1], q_waves[-1], i_waves[0]]
)  # Replaces previous wfm.
monitor.add_note("wave_1", "re-writen")

monitor.remove_wfm("wave_10")
# monitor.echo()

monitor.close()
