

import cmath
import math
import random
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")  # draw plots into files, no window needed
import matplotlib.pyplot as plt


# Helper functions (used by several versions)


def next_power_of_two(number):
    """Return the smallest power of two that is >= number."""
    power = 1
    while power < number:
        power = power * 2
    return power


def pad_with_zeros(values, new_length):
    """Return a copy of 'values' with zeros added at the end up to new_length."""
    result = list(values)
    while len(result) < new_length:
        result.append(0)
    return result


def add_lists(first, second):
    """Add two lists of the same length, position by position."""
    result = []
    for index in range(len(first)):
        result.append(first[index] + second[index])
    return result


def subtract_lists(first, second):
    """Subtract 'second' from 'first', position by position (same length)."""
    result = []
    for index in range(len(first)):
        result.append(first[index] - second[index])
    return result


# V1 - Schoolbook nested loop   (Theta(n^2) time)


def poly_mul_v1_schoolbook(a, b):
    """Every a[i] meets every b[j]. The product goes to position i + j."""
    result = [0] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            result[i + j] = result[i + j] + a[i] * b[j]
    return result



# V2 - Naive 4-way divide and conquer   (Theta(n^2) time)


def dc4_same_length(a, b):
    """
    Multiply two lists of the same length n, where n is a power of two.
    The answer has length 2n - 1.
    """
    n = len(a)

    # Base case: one coefficient each -> one scalar multiplication.
    if n == 1:
        return [a[0] * b[0]]

    m = n // 2
    a_low = a[:m]
    a_high = a[m:]
    b_low = b[:m]
    b_high = b[m:]

    # Four recursive multiplications of half size.
    high_high = dc4_same_length(a_high, b_high)
    high_low = dc4_same_length(a_high, b_low)
    low_high = dc4_same_length(a_low, b_high)
    low_low = dc4_same_length(a_low, b_low)

    # Shift and add:  C = high_high * x^(2m) + (high_low + low_high) * x^m + low_low
    result = [0] * (2 * n - 1)
    for k in range(len(low_low)):
        result[k] = result[k] + low_low[k]
    for k in range(len(high_low)):
        result[k + m] = result[k + m] + high_low[k] + low_high[k]
    for k in range(len(high_high)):
        result[k + 2 * m] = result[k + 2 * m] + high_high[k]
    return result


def poly_mul_v2_dc4(a, b):
    """Pad both inputs to the same power-of-two length, multiply, trim the extra zeros."""
    size = next_power_of_two(max(len(a), len(b)))
    padded_a = pad_with_zeros(a, size)
    padded_b = pad_with_zeros(b, size)
    full_result = dc4_same_length(padded_a, padded_b)
    return full_result[:len(a) + len(b) - 1]


# V3 - Karatsuba 3-way divide and conquer   (Theta(n^1.585) time)


def karatsuba_same_length(a, b, cutoff):
    """
    Multiply two lists of the same power-of-two length n.
    Small inputs (n <= cutoff) use the schoolbook loop, because for tiny
    lists the extra additions and recursion cost more than they save.
    """
    n = len(a)

    if n <= cutoff:
        return poly_mul_v1_schoolbook(a, b)

    m = n // 2
    a_low = a[:m]
    a_high = a[m:]
    b_low = b[:m]
    b_high = b[m:]

    # Only THREE recursive multiplications.
    p1 = karatsuba_same_length(a_high, b_high, cutoff)        # AH * BH
    p2 = karatsuba_same_length(a_low, b_low, cutoff)          # AL * BL
    a_sum = add_lists(a_high, a_low)
    b_sum = add_lists(b_high, b_low)
    p3 = karatsuba_same_length(a_sum, b_sum, cutoff)          # (AH+AL)(BH+BL)

    # Middle term = P3 - P1 - P2  (= AH*BL + AL*BH)
    middle = subtract_lists(subtract_lists(p3, p1), p2)

    # Assemble:  C = P1 * x^(2m) + middle * x^m + P2
    result = [0] * (2 * n - 1)
    for k in range(len(p2)):
        result[k] = result[k] + p2[k]
    for k in range(len(middle)):
        result[k + m] = result[k + m] + middle[k]
    for k in range(len(p1)):
        result[k + 2 * m] = result[k + 2 * m] + p1[k]
    return result


def poly_mul_v3_karatsuba(a, b, cutoff=16):
    size = next_power_of_two(max(len(a), len(b)))
    padded_a = pad_with_zeros(a, size)
    padded_b = pad_with_zeros(b, size)
    full_result = karatsuba_same_length(padded_a, padded_b, cutoff)
    return full_result[:len(a) + len(b) - 1]


# V4 - Recursive FFT   (Theta(n log n) time)


def fft_recursive(values, invert):
    """
    Evaluate the polynomial 'values' at all N-th roots of unity (N = len(values)).
    If invert is True, use the conjugate roots (for the inverse FFT).
    The division by N for the inverse is done by the caller.
    """
    n = len(values)
    if n == 1:
        return [values[0]]

    # Split into even-index and odd-index coefficients (new lists are created).
    even_part = values[0::2]
    odd_part = values[1::2]
    y_even = fft_recursive(even_part, invert)
    y_odd = fft_recursive(odd_part, invert)

    if invert:
        sign = -1
    else:
        sign = 1

    result = [0] * n
    half = n // 2
    for k in range(half):
        angle = sign * 2 * math.pi * k / n
        omega = complex(math.cos(angle), math.sin(angle))   # omega_N^k
        twiddle = omega * y_odd[k]
        result[k] = y_even[k] + twiddle          # butterfly: top output
        result[k + half] = y_even[k] - twiddle   # butterfly: bottom output
    return result


def poly_mul_v4_fft_recursive(a, b):
    result_length = len(a) + len(b) - 1
    size = next_power_of_two(result_length)

    # 1. Forward FFT of both inputs (coefficients -> point values).
    a_points = fft_recursive(pad_with_zeros(a, size), False)
    b_points = fft_recursive(pad_with_zeros(b, size), False)

    # 2. Pointwise multiplication.
    c_points = []
    for k in range(size):
        c_points.append(a_points[k] * b_points[k])

    # 3. Inverse FFT (point values -> coefficients), divide by N, round.
    c_values = fft_recursive(c_points, True)
    result = []
    for k in range(result_length):
        result.append(round(c_values[k].real / size))
    return result



# V5 - Iterative FFT   (Theta(n log n) time)


def reverse_bits(number, bit_count):
    """Reverse the lowest 'bit_count' bits.  Example: reverse_bits(1, 3) -> 4  (001 -> 100)."""
    reversed_number = 0
    for step in range(bit_count):
        lowest_bit = number & 1                       # take the last bit
        reversed_number = (reversed_number << 1) | lowest_bit
        number = number >> 1                          # drop the last bit
    return reversed_number


def fft_iterative(values, invert):
    """
    In-place FFT: 'values' is overwritten with the result.
    Only a few single variables are used as extra memory.
    """
    n = len(values)
    bit_count = int(math.log2(n))

    # Step 1: bit-reversal permutation (swap each pair only once).
    for index in range(n):
        partner = reverse_bits(index, bit_count)
        if index < partner:
            temporary = values[index]
            values[index] = values[partner]
            values[partner] = temporary

    if invert:
        sign = -1
    else:
        sign = 1

    # Step 2: butterflies, bottom-up, block sizes 2, 4, 8, ..., n.
    block_size = 2
    while block_size <= n:
        half = block_size // 2
        angle_step = sign * 2 * math.pi / block_size
        for k in range(half):
            # Compute omega once, then reuse it in every block of this size.
            angle = angle_step * k
            omega = complex(math.cos(angle), math.sin(angle))
            for block_start in range(0, n, block_size):
                top = values[block_start + k]
                bottom = omega * values[block_start + k + half]
                values[block_start + k] = top + bottom
                values[block_start + k + half] = top - bottom
        block_size = block_size * 2


def poly_mul_v5_fft_iterative(a, b):
    result_length = len(a) + len(b) - 1
    size = next_power_of_two(result_length)

    # The padded buffers: the only big memory we use.
    a_buffer = []
    b_buffer = []
    for k in range(size):
        if k < len(a):
            a_buffer.append(complex(a[k], 0))
        else:
            a_buffer.append(0j)
        if k < len(b):
            b_buffer.append(complex(b[k], 0))
        else:
            b_buffer.append(0j)

    fft_iterative(a_buffer, False)
    fft_iterative(b_buffer, False)

    # Pointwise multiply, storing the result back into a_buffer (in place).
    for k in range(size):
        a_buffer[k] = a_buffer[k] * b_buffer[k]

    fft_iterative(a_buffer, True)

    result = []
    for k in range(result_length):
        result.append(round(a_buffer[k].real / size))
    return result



# All versions in one dictionary, so experiments can loop over them


VERSIONS = {
    "V1 Schoolbook": poly_mul_v1_schoolbook,
    "V2 4-way split": poly_mul_v2_dc4,
    "V3 Karatsuba": poly_mul_v3_karatsuba,
    "V4 Recursive FFT": poly_mul_v4_fft_recursive,
    "V5 Iterative FFT": poly_mul_v5_fft_iterative,
}


def random_polynomial(length, max_size):
    coefficients = []
    for k in range(length):
        coefficients.append(random.randint(-max_size, max_size))
    return coefficients


# Experiment 1a: differential correctness test


def experiment_correctness():
    print("=" * 60)
    print("Experiment 1a: do all 5 versions give the same answer?")
    print("=" * 60)

    # Hand-worked example from Part A first.
    a = [1, 5, 2, 3]
    b = [3, 4, 1, 2]
    for name in VERSIONS:
        answer = VERSIONS[name](a, b)
        print(f"  Part A example, {name:17}: {answer}")

    all_passed = True
    for n in [8, 64, 512, 2048]:
        a = random_polynomial(n, 100)
        b = random_polynomial(n, 100)
        expected = poly_mul_v1_schoolbook(a, b)
        for name in VERSIONS:
            answer = VERSIONS[name](a, b)
            if answer == expected:
                status = "OK"
            else:
                status = "MISMATCH"
                all_passed = False
            print(f"  n = {n:5}, {name:17}: {status}")
    if all_passed:
        print("  All versions agree.")
    else:
        print("  Some versions disagree!")
    print()

# Experiment 1b: floating-point precision stress test


def experiment_precision():
    print("=" * 60)
    print("Experiment 1b: when does float64 FFT rounding break? (n = 4096)")
    print("=" * 60)
    n = 4096
    rows = []
    for exponent in [4, 5, 6, 7, 8]:
        max_size = 10 ** exponent
        a = random_polynomial(n, max_size)
        b = random_polynomial(n, max_size)
        exact = poly_mul_v1_schoolbook(a, b)          # Python ints: always exact

        largest_value = 0
        for value in exact:
            if abs(value) > largest_value:
                largest_value = abs(value)

        for name in ["V4 Recursive FFT", "V5 Iterative FFT"]:
            answer = VERSIONS[name](a, b)
            wrong = 0
            biggest_error = 0
            for k in range(len(exact)):
                error = abs(answer[k] - exact[k])
                if error != 0:
                    wrong = wrong + 1
                if error > biggest_error:
                    biggest_error = error
            rows.append((exponent, name, largest_value, wrong, biggest_error))
            print(f"  coeffs up to 10^{exponent}, {name:17}: "
                  f"largest |c_k| = {largest_value:.2e}, "
                  f"wrong coefficients = {wrong:5} of {len(exact)}, "
                  f"max error = {biggest_error}")
    print(f"  For reference: 2^53 = {2 ** 53:.2e} (largest exactly stored integer in float64)")
    print()
    return rows


# Experiment 2: runtime and memory benchmark

TIME_LIMIT_SECONDS = 8.0   # stop testing a version once one run takes longer than this


def time_one_run(function, a, b):
    start = time.perf_counter()
    function(a, b)
    return time.perf_counter() - start


def best_time(function, a, b, n):
    """Run a few times for small n (timings are noisy) and keep the fastest."""
    if n <= 1024:
        repeats = 5
    else:
        repeats = 1
    best = None
    for attempt in range(repeats):
        seconds = time_one_run(function, a, b)
        if best is None or seconds < best:
            best = seconds
    return best


def peak_memory(function, a, b):
    tracemalloc.start()
    function(a, b)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def experiment_benchmark():
    print("=" * 60)
    print("Experiment 2: runtime and peak memory for n = 2^4 ... 2^15")
    print("=" * 60)
    sizes = []
    for power in range(4, 16):
        sizes.append(2 ** power)

    times = {}
    memory = {}
    stopped = {}
    for name in VERSIONS:
        times[name] = {}
        memory[name] = {}
        stopped[name] = False

    for n in sizes:
        a = random_polynomial(n, 100)
        b = random_polynomial(n, 100)
        for name in VERSIONS:
            if stopped[name]:
                continue
            function = VERSIONS[name]
            seconds = best_time(function, a, b, n)
            times[name][n] = seconds
            # tracemalloc slows code down, so memory is measured in a separate run.
            if seconds < TIME_LIMIT_SECONDS / 2:
                memory[name][n] = peak_memory(function, a, b)
            print(f"  n = {n:6}, {name:17}: {seconds:9.4f} s")
            if seconds > TIME_LIMIT_SECONDS:
                stopped[name] = True
                print(f"      ({name} is now too slow; skipping larger n)")
    print()
    return sizes, times, memory


def find_crossover(times, slower_name, faster_name):
    """Smallest n from which 'faster_name' beats 'slower_name' for every larger n tested."""
    shared_sizes = []
    for n in sorted(times[slower_name]):
        if n in times[faster_name]:
            shared_sizes.append(n)
    crossover = None
    for n in reversed(shared_sizes):
        if times[faster_name][n] < times[slower_name][n]:
            crossover = n
        else:
            break
    return crossover


# Plots

COLORS = {
    "V1 Schoolbook": "#2a78d6",
    "V2 4-way split": "#eb6834",
    "V3 Karatsuba": "#1baf7a",
    "V4 Recursive FFT": "#eda100",
    "V5 Iterative FFT": "#e87ba4",
}
MARKERS = {
    "V1 Schoolbook": "o",
    "V2 4-way split": "s",
    "V3 Karatsuba": "^",
    "V4 Recursive FFT": "D",
    "V5 Iterative FFT": "v",
}


def draw_plot(data, y_label, title, file_name, divide_by):
    figure, axes = plt.subplots(figsize=(10, 6))
    for name in VERSIONS:
        x_values = []
        y_values = []
        for n in sorted(data[name]):
            x_values.append(n)
            y_values.append(data[name][n] / divide_by)
        if len(x_values) == 0:
            continue
        axes.plot(x_values, y_values, label=name, color=COLORS[name],
                  marker=MARKERS[name], markersize=8, linewidth=2)
    axes.set_xscale("log", base=2)
    axes.set_yscale("log")
    axes.set_xlabel("n (number of coefficients per polynomial)")
    axes.set_ylabel(y_label)
    axes.set_title(title, loc="left", fontsize=13)
    axes.grid(True, which="major", color="#e6e4dd", linewidth=0.8)
    for side in ["top", "right"]:
        axes.spines[side].set_visible(False)
    axes.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(file_name, dpi=150)
    plt.close(figure)
    print(f"  Saved {file_name}")


# Main program

def main():
    random.seed(7)  # same random numbers every run, so results are repeatable
    experiment_correctness()
    experiment_precision()
    sizes, times, memory = experiment_benchmark()

    karatsuba_point = find_crossover(times, "V1 Schoolbook", "V3 Karatsuba")
    fft_point = find_crossover(times, "V3 Karatsuba", "V5 Iterative FFT")
    print("Crossover results:")
    print(f"  Karatsuba beats Schoolbook from n = {karatsuba_point}")
    print(f"  Iterative FFT beats Karatsuba from n = {fft_point}")
    print()

    draw_plot(times, "wall-clock time (seconds, log scale)",
              "Runtime of the five polynomial multiplication versions",
              "runtime_plot.png", 1)
    draw_plot(memory, "peak memory (KiB, log scale)",
              "Peak memory (tracemalloc) of the five versions",
              "memory_plot.png", 1024)


if __name__ == "__main__":
    main()
