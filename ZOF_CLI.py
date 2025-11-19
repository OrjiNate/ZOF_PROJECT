import math
import sys
from typing import Callable, Tuple

# Build safe eval environment for math functions
_safe_dict = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
_safe_dict.update({
    'abs': abs, 'min': min, 'max': max, 'pow': pow
})

def parse_function(expr: str) -> Callable[[float], float]:
    """Return a function f(x) evaluating the expression using math functions."""
    # allow 'x' and math functions
    def f(x):
        try:
            return float(eval(expr, {"__builtins__": None}, dict(_safe_dict, x=x)))
        except Exception as e:
            raise ValueError(f"Error evaluating function at x={x}: {e}")
    return f

def numeric_derivative(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """Central difference derivative."""
    return (f(x + h) - f(x - h)) / (2 * h)

def print_iteration_header():
    print(f"{'Iter':>4}  {'x_n':>14}  {'f(x_n)':>14}  {'Error':>12}")

def bisection(f: Callable[[float], float], a: float, b: float, tol: float, max_iter: int):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Bisection.")
    prev_c = None
    print_iteration_header()
    for i in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        err = abs(c - prev_c) if prev_c is not None else float('nan')
        print(f"{i:4d}  {c:14.8f}  {fc:14.8e}  {err:12.8e}")
        if abs(fc) < 1e-16 or (prev_c is not None and abs(c - prev_c) < tol) or (b - a)/2 < tol:
            return c, abs(c - (prev_c if prev_c is not None else c)), i
        prev_c = c
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c, abs(c - prev_c), max_iter

def regula_falsi(f: Callable[[float], float], a: float, b: float, tol: float, max_iter: int):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Regula Falsi.")
    prev_c = None
    print_iteration_header()
    for i in range(1, max_iter + 1):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        err = abs(c - prev_c) if prev_c is not None else float('nan')
        print(f"{i:4d}  {c:14.8f}  {fc:14.8e}  {err:12.8e}")
        if abs(fc) < 1e-16 or (prev_c is not None and abs(c - prev_c) < tol):
            return c, abs(c - (prev_c if prev_c is not None else c)), i
        prev_c = c
        # Update endpoints (classic Regula Falsi)
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c, abs(c - prev_c), max_iter

def secant(f: Callable[[float], float], x0: float, x1: float, tol: float, max_iter: int):
    prev = x0
    curr = x1
    print_iteration_header()
    for i in range(1, max_iter + 1):
        f_prev, f_curr = f(prev), f(curr)
        if f_curr - f_prev == 0:
            raise ZeroDivisionError("Division by zero in Secant method (f(x1)-f(x0)=0).")
        next_x = curr - f_curr * (curr - prev) / (f_curr - f_prev)
        err = abs(next_x - curr)
        print(f"{i:4d}  {curr:14.8f}  {f_curr:14.8e}  {err:12.8e}")
        if err < tol:
            return next_x, err, i
        prev, curr = curr, next_x
    return curr, abs(curr - prev), max_iter

def newton_raphson(f: Callable[[float], float], x0: float, tol: float, max_iter: int):
    x = x0
    print_iteration_header()
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = numeric_derivative(f, x)
        if dfx == 0:
            raise ZeroDivisionError("Zero derivative encountered in Newton-Raphson.")
        x_next = x - fx / dfx
        err = abs(x_next - x)
        print(f"{i:4d}  {x:14.8f}  {fx:14.8e}  {err:12.8e}")
        if err < tol:
            return x_next, err, i
        x = x_next
    return x, err, max_iter

def fixed_point(g: Callable[[float], float], x0: float, tol: float, max_iter: int):
    x = x0
    print_iteration_header()
    for i in range(1, max_iter + 1):
        x_next = g(x)
        fx = None
        err = abs(x_next - x)
        # We can't compute f(x) here (we only have g). Show x and error.
        print(f"{i:4d}  {x_next:14.8f}  {'-':>14}  {err:12.8e}")
        if err < tol:
            return x_next, err, i
        x = x_next
    return x, err, max_iter

def modified_secant(f: Callable[[float], float], x0: float, delta: float, tol: float, max_iter: int):
    x = x0
    print_iteration_header()
    for i in range(1, max_iter + 1):
        fx = f(x)
        denom = f(x + delta * x) - fx
        if denom == 0:
            raise ZeroDivisionError("Division by zero in Modified Secant (denominator=0).")
        x_next = x - fx * (delta * x) / denom
        err = abs(x_next - x)
        print(f"{i:4d}  {x:14.8f}  {fx:14.8e}  {err:12.8e}")
        if err < tol:
            return x_next, err, i
        x = x_next
    return x, err, max_iter

def get_float(prompt: str) -> float:
    while True:
        try:
            val = float(input(prompt))
            return val
        except Exception as e:
            print("Invalid float. Try again.")

def get_int(prompt: str, default=None) -> int:
    while True:
        try:
            s = input(prompt)
            if s.strip() == "" and default is not None:
                return default
            return int(s)
        except Exception:
            print("Invalid integer. Try again.")

def main_menu():
    menu = """
ZOF - Zero of Functions Solver (CLI)
Choose a method:
1. Bisection Method
2. Regula Falsi (False Position)
3. Secant Method
4. Newton–Raphson Method
5. Fixed Point Iteration (requires g(x))
6. Modified Secant Method
0. Exit
"""
    print(menu)
    choice = input("Enter choice (0-6): ").strip()
    return choice

def run():
    while True:
        choice = main_menu()
        if choice == '0':
            print("Exiting.")
            sys.exit(0)

        if choice not in {'1','2','3','4','5','6'}:
            print("Invalid choice. Try again.")
            continue

        # For methods 1-4 & 6 we need f(x). For 5 we need g(x).
        if choice == '5':
            print("Enter g(x) for Fixed Point Iteration (x_{n+1} = g(x_n)).")
            g_expr = input("g(x) = ").strip()
            try:
                g = parse_function(g_expr)
            except Exception as e:
                print(f"Failed to parse g(x): {e}")
                continue
        else:
            print("Enter the function f(x) as a Python expression using math functions (e.g. 'x**3 - x - 2').")
            print("You may use math functions like sin(x), cos(x), exp(x), log(x), etc.")
            f_expr = input("f(x) = ").strip()
            try:
                f = parse_function(f_expr)
            except Exception as e:
                print(f"Failed to parse f(x): {e}")
                continue

        tol = float(input("Tolerance (e.g. 1e-6): ") or 1e-6)
        max_iter = get_int("Max iterations (e.g. 50): ", default=50)

        try:
            if choice == '1':  # Bisection
                a = get_float("Left endpoint a: ")
                b = get_float("Right endpoint b: ")
                root, final_err, iters = bisection(f, a, b, tol, max_iter)
            elif choice == '2':  # Regula Falsi
                a = get_float("Left endpoint a: ")
                b = get_float("Right endpoint b: ")
                root, final_err, iters = regula_falsi(f, a, b, tol, max_iter)
            elif choice == '3':  # Secant
                x0 = get_float("Initial guess x0: ")
                x1 = get_float("Initial guess x1: ")
                root, final_err, iters = secant(f, x0, x1, tol, max_iter)
            elif choice == '4':  # Newton-Raphson
                x0 = get_float("Initial guess x0: ")
                root, final_err, iters = newton_raphson(f, x0, tol, max_iter)
            elif choice == '5':  # Fixed Point Iteration
                x0 = get_float("Initial guess x0: ")
                root, final_err, iters = fixed_point(g, x0, tol, max_iter)
            elif choice == '6':  # Modified Secant
                x0 = get_float("Initial guess x0: ")
                delta = float(input("Delta (fraction, e.g. 1e-3): ") or 1e-3)
                root, final_err, iters = modified_secant(f, x0, delta, tol, max_iter)
            else:
                print("Invalid option.")
                continue
        except Exception as e:
            print(f"Computation error: {e}")
            continue

        print("\nResult Summary")
        print("--------------")
        print(f"Estimated root: {root:.12g}")
        print(f"Final error estimate: {final_err:.8e}")
        print(f"Iterations: {iters}")
        print("--------------\n")

        cont = input("Do you want to solve another problem? (y/n): ").strip().lower()
        if cont != 'y':
            print("Goodbye.")
            break

if __name__ == "__main__":
    run()
