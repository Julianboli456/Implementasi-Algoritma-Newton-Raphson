"""Newton-Raphson dan biseksi. Python 3, tanpa paket tambahan."""
import math

def newton(f, df, x0, tol_x=1e-8, tol_f=1e-8, max_iter=100):
    if tol_x <= 0 or tol_f <= 0 or max_iter < 1:
        raise ValueError("Toleransi dan batas iterasi harus positif.")
    x = float(x0)
    if not math.isfinite(x):
        raise ValueError("Tebakan awal harus berhingga.")
    fx = f(x)
    if not math.isfinite(fx):
        raise ValueError("Nilai fungsi tidak berhingga.")
    history = [(0, x, abs(fx), None)]
    if abs(fx) <= tol_f:
        return x, history
    for k in range(1, max_iter + 1):
        d = df(x)
        if not math.isfinite(d) or abs(d) < 1e-12:
            raise ValueError("Turunan nol atau terlalu kecil.")
        xn = x - fx / d
        if not math.isfinite(xn):
            raise ValueError("Iterasi tidak berhingga.")
        fn = f(xn)
        if not math.isfinite(fn):
            raise ValueError("Nilai fungsi tidak berhingga.")
        step = abs(xn - x)
        history.append((k, xn, abs(fn), step))
        if step <= tol_x and abs(fn) <= tol_f:
            return xn, history
        x, fx = xn, fn
    raise RuntimeError("Belum konvergen dalam batas iterasi.")

def bisection(f, a, b, tol_x=1e-8, tol_f=1e-8, max_iter=100):
    if not a < b or tol_x <= 0 or tol_f <= 0 or max_iter < 1:
        raise ValueError("Interval atau parameter tidak valid.")
    fa, fb = f(a), f(b)
    if not all(math.isfinite(v) for v in (a, b, fa, fb)):
        raise ValueError("Input tidak berhingga.")
    if abs(fa) <= tol_f:
        return a, [(0, a, abs(fa), 0.0)]
    if abs(fb) <= tol_f:
        return b, [(0, b, abs(fb), 0.0)]
    if fa * fb > 0:
        raise ValueError("Interval harus mengapit akar.")
    history = []
    for k in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        if not math.isfinite(fc):
            raise ValueError("Nilai fungsi tidak berhingga.")
        half_width = (b - a) / 2
        history.append((k, c, abs(fc), half_width))
        if fc == 0 or (half_width <= tol_x and abs(fc) <= tol_f):
            return c, history
        if fa * fc < 0:
            b = c
        else:
            a, fa = c, fc
    raise RuntimeError("Belum konvergen dalam batas iterasi.")

CASES = [
    {"name": "Sisi taman persegi", "unit": "m",
     "f": lambda x: x*x - 50, "df": lambda x: 2*x,
     "x0": 7., "interval": (7., 8.), "exact": math.sqrt(50)},
    {"name": "Rusuk bak kubus", "unit": "m",
     "f": lambda x: x**3 - 10, "df": lambda x: 3*x*x,
     "x0": 2., "interval": (2., 3.), "exact": 10**(1/3)},
    {"name": "Waktu benda mencapai tanah", "unit": "s",
     "f": lambda t: 20 + 5*t - 5*t*t,
     "df": lambda t: 5 - 10*t,
     "x0": 3., "interval": (2., 3.),
     "exact": (1 + math.sqrt(17))/2},
]

if __name__ == "__main__":
    for case in CASES:
        root, rows = newton(case["f"], case["df"], case["x0"])
        print("\n" + case["name"])
        print(" k       pendekatan          residu        perubahan")
        for k, x, residual, step in rows:
            delta = "-" if step is None else f"{step:.3e}"
            print(f"{k:2d} {x:17.10f} {residual:14.3e} {delta:>14}")
        print(f"Hasil: {root:.10f} {case['unit']}")
        rb, hb = bisection(case["f"], *case["interval"])
        print(f"Biseksi: {rb:.10f}; iterasi: {hb[-1][0]}")
