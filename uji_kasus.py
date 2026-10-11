"""Jalankan: python uji_kasus.py."""
import math
import unittest
from newton_raphson import newton, bisection, CASES

class Tests(unittest.TestCase):
    def test_three_cases(self):
        for c in CASES:
            with self.subTest(case=c["name"]):
                r, h = newton(c["f"], c["df"], c["x0"])
                self.assertLess(abs(r-c["exact"]), 1e-8)
                self.assertLessEqual(h[-1][2], 1e-8)
                self.assertLessEqual(h[-1][3], 1e-8)
                rb, _ = bisection(c["f"], *c["interval"])
                self.assertLess(abs(rb-c["exact"]), 1e-8)
    def test_zero_derivative(self):
        with self.assertRaises(ValueError):
            newton(lambda x:x*x-50, lambda x:2*x, 0)
    def test_cycle(self):
        with self.assertRaises(RuntimeError):
            newton(lambda x:x**3-2*x+2, lambda x:3*x*x-2, 0, max_iter=10)
    def test_initial_root(self):
        r,h = newton(lambda x:x*x, lambda x:2*x, 0)
        self.assertEqual((r,len(h)), (0,1))
    def test_invalid_bracket(self):
        with self.assertRaises(ValueError):
            bisection(lambda x:x*x+1, -1, 1)
    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            newton(lambda x:x, lambda x:1, math.inf)
    def test_iteration_limit(self):
        with self.assertRaises(RuntimeError):
            newton(lambda x:x*x-50, lambda x:2*x, 7, max_iter=1)

if __name__ == "__main__":
    unittest.main(verbosity=2)
