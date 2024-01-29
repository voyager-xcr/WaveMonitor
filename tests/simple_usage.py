from wave_monitor import WaveMonitor
import numpy as np

monitor = WaveMonitor()
monitor.run_monitor_window("DEBUG")
monitor.autoscale()
# monitor.clear()

t = np.linspace(0, 1, 1_000_001)  # 1m pts ~= 1ms for 1GSa/s.
n = 20  # Okay with 20.
i_waves = [np.cos(2 * np.pi * f * t)[:-f] for f in range(1, n + 1)]
q_waves = [np.sin(2 * np.pi * f * t)[:-f] for f in range(1, n + 1)]

for i, (i_wave, q_wave) in enumerate(zip(i_waves, q_waves)):
    monitor.add_wfm(f"wave_{i%15}", t[:i_wave.size], [i_wave, q_wave])
monitor.autoscale()

monitor.remove_wfm("wave_10")
monitor.echo()
