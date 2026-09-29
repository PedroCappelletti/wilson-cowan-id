#!/usr/bin/env python3
# Pide a Windows que no suspenda la máquina mientras haya corridas andando.
#
# El 28-09 la máquina se durmió de 20:59 a 23:40 con cuatro corridas en curso:
# el pedido de mantenerla despierta de la app se suelta cuando la sesión queda
# inactiva. Esto hace lo mismo que un reproductor de video: una solicitud de
# energía que dura lo que dura este proceso. No toca ninguna configuración, y
# con la tapa cerrada la máquina se duerme igual.
#
#   python scripts/despierto.py      # sale solo cuando no queda ninguna corrida

from __future__ import annotations

import ctypes
import subprocess
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

CONTAR = ("(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -match 'esc_run|cola_' }).Count")


def corridas_vivas() -> int:
    r = subprocess.run(["powershell", "-NoProfile", "-Command", CONTAR],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip() or 0)
    except ValueError:
        return 1   # ante la duda, seguir despierto


def main():
    if sys.platform != "win32":
        return
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    print(f"{time.strftime('%H:%M')} pedido de no suspender activo", flush=True)
    while corridas_vivas() > 0:
        time.sleep(60)
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print(f"{time.strftime('%H:%M')} no quedan corridas, pedido liberado", flush=True)


if __name__ == "__main__":
    main()
