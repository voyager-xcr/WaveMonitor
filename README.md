# Wave Monitor

A simple GUI for monitoring waveforms. It plots waveforms with PyQtGraph in a separate process. The GUI is built with PySide6.

The `WaveMonitor` class is the main interface. It provides methods for adding and removing waveforms from the plot, clearing the plot, and etc.

In GUI, right click to show the menu. Keyboard shortcuts are also supported.

# Installation

```bash
pip install wave-monitor
```

or install from source.

```bash
pip install git+https://github.com/Qiujv/LabCodes.git
```

# Usage
Avoid calling `clear` if you only want to update the plot. It is more efficient to update the plot with `add_wfm`.

```python
from wave_monitor import WaveMonitor
import numpy as np

monitor = WaveMonitor()
monitor.find_or_run_monitor_window("DEBUG")
monitor.autoscale()
# monitor.clear()

t = np.linspace(0, 1, 1_000_001)  # 1m pts ~= 1ms for 1GSa/s.
n = 20  # Okay with 20.
i_waves = [np.cos(2 * np.pi * f * t) for f in range(1, n + 1)]
q_waves = [np.sin(2 * np.pi * f * t) for f in range(1, n + 1)]

for i, (i_wave, q_wave) in enumerate(zip(i_waves, q_waves)):
    monitor.add_wfm(f"wave_{i%15}", t, [i_wave, q_wave])
monitor.autoscale()

monitor.remove_wfm("wave_10")
monitor.echo()

```

# Thanks

This project is derived from [WaveViewer](https://github.com/kahojyun/wave-viewer).
