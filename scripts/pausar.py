#!/usr/bin/env python3
# Congela o reanuda las corridas de esc_run.py sin perderlas.
#
# Una corrida no guarda estado a mitad de camino: si se mata, se pierde entera.
# Congelarla la deja en memoria sin usar CPU, y sobrevive a una suspensión de
# la máquina (no a un apagado). Usa NtSuspendProcess / NtResumeProcess de
# Windows sobre cada proceso de esc_run.py.
#
#   python scripts/pausar.py              congela
#   python scripts/pausar.py --reanudar   reanuda

from __future__ import annotations

import ctypes
import json
import subprocess
import sys

PS = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
      "Where-Object { $_.CommandLine -like '*esc_run.py*' } | "
      "Select-Object ProcessId, CommandLine | ConvertTo-Json")


def corridas():
    out = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                         capture_output=True, text=True).stdout.strip()
    if not out:
        return []
    datos = json.loads(out)
    datos = datos if isinstance(datos, list) else [datos]
    return [(d["ProcessId"], d["CommandLine"].split("--tag ")[-1]) for d in datos]


def aplicar(reanudar: bool):
    ntdll = ctypes.WinDLL("ntdll")
    k32 = ctypes.WinDLL("kernel32")
    PROCESS_SUSPEND_RESUME = 0x0800
    f = ntdll.NtResumeProcess if reanudar else ntdll.NtSuspendProcess
    for pid, tag in corridas():
        h = k32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
        if not h:
            print(f"  no pude abrir {tag} ({pid})")
            continue
        st = f(h)
        k32.CloseHandle(h)
        print(f"  {'reanudada' if reanudar else 'congelada'} {tag} ({pid})"
              + ("" if st == 0 else f", estado {st:#x}"))


if __name__ == "__main__":
    aplicar("--reanudar" in sys.argv)
