from pathlib import Path
import os
import sys
import multiprocessing
import traceback

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

def check_runtime():
    import inspect
    import numpy, pandas, matplotlib, tkinter
    from scipy.optimize import least_squares
    if 'callback' not in inspect.signature(least_squares).parameters:
        raise RuntimeError('SciPy 1.16 or later is required. See 03_Bin/README.md.')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / '03_Bin' / 'src'))
    try:
        check_runtime()
        if sys.argv[1:] == ['--check']:
            sys.exit(0)
        if len(sys.argv) > 1:
            from eis_fitting.cli import main
        else:
            from eis_fitting.desktop import main
        main(root)
    except Exception:
        error = traceback.format_exc()
        if len(sys.argv) > 1:
            print(error, file=sys.stderr)
        else:
            import tkinter as tk
            from tkinter import messagebox
            win = tk.Tk()
            win.withdraw()
            messagebox.showerror('EIS Fitting', error[-2500:])
            win.destroy()
        sys.exit(1)
