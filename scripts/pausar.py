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
#   python scripts/pausar.py --tag X      congela solo X, la anota en
#                                         logs/escalado/PAUSADAS y le saca la
#                                         memoria a disco; la cola (cola.py) no la
#                                         cuenta como lugar ocupado y la reanuda
#                                         cuando se libera uno

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
from pathlib import Path

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


PAUSADAS = Path(__file__).resolve().parents[1] / "logs" / "escalado" / "PAUSADAS"


def cambiar(pid: int, tag: str, reanudar: bool, a_disco: bool = False) -> bool:
    ntdll = ctypes.WinDLL("ntdll")
    k32 = ctypes.WinDLL("kernel32")
    # SUSPEND_RESUME, y para vaciar la memoria SET_QUOTA y QUERY_INFORMATION
    h = k32.OpenProcess(0x0800 | 0x0100 | 0x0400, False, pid)
    if not h:
        print(f"  no pude abrir {tag} ({pid})")
        return False
    st = (ntdll.NtResumeProcess if reanudar else ntdll.NtSuspendProcess)(h)
    # Congelada, su memoria sigue reservada hasta que Windows la necesite.
    # Vaciar el conjunto de trabajo la manda al archivo de paginación ya.
    if a_disco and not reanudar:
        ctypes.WinDLL("psapi").EmptyWorkingSet(h)
    k32.CloseHandle(h)
    print(f"  {'reanudada' if reanudar else 'congelada'} {tag} ({pid})"
          + ("" if st == 0 else f", estado {st:#x}"))
    return st == 0


def pausadas() -> dict[str, int]:
    if not PAUSADAS.exists():
        return {}
    return {t: int(p) for t, p in (l.split() for l in PAUSADAS.read_text().splitlines() if l.strip())}


def anotar(d: dict[str, int]) -> None:
    PAUSADAS.write_text("".join(f"{t} {p}\n" for t, p in d.items()))


def aplicar(reanudar: bool, solo: str | None = None):
    for pid, tag in corridas():
        if solo and tag != solo:
            continue
        if cambiar(pid, tag, reanudar, a_disco=bool(solo)) and solo:
            d = pausadas()
            d.pop(tag, None) if reanudar else d.__setitem__(tag, pid)
            anotar(d)


if __name__ == "__main__":
    solo = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else None
    aplicar("--reanudar" in sys.argv, solo)
