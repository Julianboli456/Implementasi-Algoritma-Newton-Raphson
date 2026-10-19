"""Newton-Raphson dan biseksi. Python 3, tanpa paket tambahan."""
import math


def newton(f, df, x0, tol_x=1e-8, tol_f=1e-8, max_iter=100):
    """Mencari akar f(x) = 0 dengan metode Newton-Raphson.

    Rumus iterasi: x_{n+1} = x_n - f(x_n) / f'(x_n)

    Parameter:
        f        : fungsi yang akarnya dicari
        df       : turunan pertama f
        x0       : tebakan awal
        tol_x    : toleransi untuk perubahan antar-iterasi |x_{n+1} - x_n|
        tol_f    : toleransi untuk residu |f(x)|
        max_iter : batas jumlah iterasi

    Mengembalikan (akar, riwayat). Setiap baris riwayat berbentuk
    (k, x, |f(x)|, |perubahan x|); perubahan bernilai None pada k = 0.
    """
    # Validasi parameter: toleransi dan batas iterasi harus positif
    if tol_x <= 0 or tol_f <= 0 or max_iter < 1:
        raise ValueError("Toleransi dan batas iterasi harus positif.")

    # Tebakan awal diubah ke float dan harus berhingga (bukan inf/NaN)
    x = float(x0)
    if not math.isfinite(x):
        raise ValueError("Tebakan awal harus berhingga.")

    # Hitung f di tebakan awal dan pastikan hasilnya sah
    fx = f(x)
    if not math.isfinite(fx):
        raise ValueError("Nilai fungsi tidak berhingga.")

    # Iterasi ke-0 dicatat sebagai titik awal (belum ada "perubahan")
    history = [(0, x, abs(fx), None)]

    # Kasus khusus: tebakan awal sudah merupakan akar
    if abs(fx) <= tol_f:
        return x, history

    for k in range(1, max_iter + 1):
        d = df(x)
        # Turunan nol/terlalu kecil membuat langkah f/d meledak atau
        # tak terdefinisi (garis singgung hampir mendatar)
        if not math.isfinite(d) or abs(d) < 1e-12:
            raise ValueError("Turunan nol atau terlalu kecil.")

        # Langkah Newton: titik potong garis singgung dengan sumbu-x
        xn = x - fx / d
        if not math.isfinite(xn):
            raise ValueError("Iterasi tidak berhingga.")

        fn = f(xn)
        if not math.isfinite(fn):
            raise ValueError("Nilai fungsi tidak berhingga.")

        step = abs(xn - x)  # besar perubahan pendekatan pada iterasi ini
        history.append((k, xn, abs(fn), step))

        # Berhenti hanya jika KEDUA kriteria terpenuhi: langkah kecil
        # DAN residu kecil (lebih andal daripada salah satu saja)
        if step <= tol_x and abs(fn) <= tol_f:
            return xn, history

        # Siapkan iterasi berikutnya
        x, fx = xn, fn

    # Habis iterasi tanpa memenuhi kriteria (mis. divergen atau berputar)
    raise RuntimeError("Belum konvergen dalam batas iterasi.")


def bisection(f, a, b, tol_x=1e-8, tol_f=1e-8, max_iter=100):
    """Mencari akar f(x) = 0 pada interval [a, b] dengan metode biseksi.

    Syarat: f(a) dan f(b) berlawanan tanda sehingga ada akar di dalam
    interval (Teorema Nilai Antara). Interval dibagi dua berulang kali
    dan separuh yang masih mengapit akar dipertahankan.

    Setiap baris riwayat berbentuk (k, c, |f(c)|, setengah lebar interval).
    """
    # Validasi: interval harus a < b, parameter harus positif
    if not a < b or tol_x <= 0 or tol_f <= 0 or max_iter < 1:
        raise ValueError("Interval atau parameter tidak valid.")

    fa, fb = f(a), f(b)
    # Semua masukan dan nilai fungsi di ujung harus berhingga
    if not all(math.isfinite(v) for v in (a, b, fa, fb)):
        raise ValueError("Input tidak berhingga.")

    # Kasus khusus: salah satu ujung interval sudah merupakan akar
    if abs(fa) <= tol_f:
        return a, [(0, a, abs(fa), 0.0)]
    if abs(fb) <= tol_f:
        return b, [(0, b, abs(fb), 0.0)]

    # Tanda sama di kedua ujung berarti akar tidak dijamin ada di dalam
    if fa * fb > 0:
        raise ValueError("Interval harus mengapit akar.")

    history = []
    for k in range(1, max_iter + 1):
        c = (a + b) / 2  # titik tengah interval
        fc = f(c)
        if not math.isfinite(fc):
            raise ValueError("Nilai fungsi tidak berhingga.")

        # Setengah lebar interval = batas galat pendekatan c
        half_width = (b - a) / 2
        history.append((k, c, abs(fc), half_width))

        # Berhenti jika tepat nol, atau interval dan residu sama-sama kecil
        if fc == 0 or (half_width <= tol_x and abs(fc) <= tol_f):
            return c, history

        # Pertahankan separuh interval yang masih mengapit akar
        if fa * fc < 0:
            b = c            # akar di [a, c]
        else:
            a, fa = c, fc    # akar di [c, b]; fa diperbarui ke f(c)

    raise RuntimeError("Belum konvergen dalam batas iterasi.")


# Tiga kasus uji berbasis soal cerita. Setiap kasus berisi:
#   name     : nama masalah
#   unit     : satuan hasil
#   f, df    : fungsi dan turunannya
#   x0       : tebakan awal untuk Newton-Raphson
#   interval : interval pengapit akar untuk biseksi
#   exact    : nilai akar eksak (pembanding)
CASES = [
    # Luas taman 50 m^2  ->  x^2 = 50  ->  x = sqrt(50)
    {"name": "Sisi taman persegi", "unit": "m",
     "f": lambda x: x*x - 50, "df": lambda x: 2*x,
     "x0": 7., "interval": (7., 8.), "exact": math.sqrt(50)},
    # Volume bak 10 m^3  ->  x^3 = 10  ->  x = 10^(1/3)
    {"name": "Rusuk bak kubus", "unit": "m",
     "f": lambda x: x**3 - 10, "df": lambda x: 3*x*x,
     "x0": 2., "interval": (2., 3.), "exact": 10**(1/3)},
    # Tinggi h(t) = 20 + 5t - 5t^2 = 0 (tanah). Akar positif dari rumus
    # kuadrat: t = (1 + sqrt(17)) / 2
    {"name": "Waktu benda mencapai tanah", "unit": "s",
     "f": lambda t: 20 + 5*t - 5*t*t,
     "df": lambda t: 5 - 10*t,
     "x0": 3., "interval": (2., 3.),
     "exact": (1 + math.sqrt(17))/2},
]

# Bagian ini hanya berjalan bila file dieksekusi langsung,
# bukan saat di-import (mis. oleh uji_kasus.py)
if __name__ == "__main__":
    for case in CASES:
        root, rows = newton(case["f"], case["df"], case["x0"])
        print("\n" + case["name"])
        print(" k       pendekatan          residu        perubahan")
        for k, x, residual, step in rows:
            # Iterasi ke-0 tidak punya perubahan, tampilkan "-"
            delta = "-" if step is None else f"{step:.3e}"
            print(f"{k:2d} {x:17.10f} {residual:14.3e} {delta:>14}")
        print(f"Hasil: {root:.10f} {case['unit']}")

        # Bandingkan dengan biseksi: lebih lambat tetapi pasti konvergen
        rb, hb = bisection(case["f"], *case["interval"])
        print(f"Biseksi: {rb:.10f}; iterasi: {hb[-1][0]}")
