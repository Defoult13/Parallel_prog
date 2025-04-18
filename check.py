import numpy as np
import matplotlib.pyplot as plt
import os
import re

INPUT_DIR = "input"
NUM_FILES_PER_SIZE = 10

def read_matrix(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        matrix = [list(map(float, line.strip().split())) for line in lines]
    return np.array(matrix)


def verify_multiplication(size, log_file=None):
    all_correct = True

    for i in range(1, NUM_FILES_PER_SIZE + 1):
        filename_A = os.path.join(INPUT_DIR, f"MatrixA({size})_{i}.txt")
        filename_B = os.path.join(INPUT_DIR, f"MatrixB({size})_{i}.txt")
        filename_Result = os.path.join(INPUT_DIR, f"Result({size})_{i}.txt")

        if not all(os.path.exists(f) for f in [filename_A, filename_B, filename_Result]):
            message = f"[Пропущено] Файлы MatrixA/MatrixB/Result для размерности {size}, набор {i} не найдены."
            print(message)
            if log_file:
                log_file.write(message + "\n")
            all_correct = False
            continue

        A = read_matrix(filename_A)
        B = read_matrix(filename_B)
        Result = read_matrix(filename_Result)
        calculated = np.matmul(A, B)

        if np.allclose(calculated, Result, atol=1e-6):
            message = f"[OK] Умножение для {size}, набор {i} корректно."
        else:
            message = f"[Ошибка] Умножение для {size}, набор {i} некорректно."
            all_correct = False

        print(message)
        if log_file:
            log_file.write(message + "\n")

    return all_correct


def parse_timing_results(filename):
    sizes = []
    times = []
    with open(filename, 'r', encoding='cp1251') as f:
        lines = f.readlines()[1:]
        for line in lines:
            match = re.match(r"(\d+x\d+)\s+([0-9.]+)", line.strip())
            if match:
                size, time = match.groups()
                sizes.append(size)
                times.append(float(time))
    return sizes, times


def plot_timing(sizes, times):
    numeric_sizes = [int(s.split('x')[0]) for s in sizes]
    plt.figure(figsize=(10, 6))
    plt.plot(numeric_sizes, times, marker='o', color='blue')
    plt.title("Зависимость времени перемножения от размерности матриц")
    plt.xlabel("Размерность матрицы (N x N)")
    plt.ylabel("Среднее время (сек)")
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join("timing_plot.png")
    plt.savefig(output_path, dpi=300)
    print(f"График сохранён в файл: {output_path}")

    plt.show()


def main():
    timing_file = os.path.join(INPUT_DIR, "timing_results.txt")

    if not os.path.exists(timing_file):
        print(f"Файл {timing_file} не найден.")
        return

    timing_sizes, times = parse_timing_results(timing_file)
    manual_sizes = [int(s.split('x')[0]) for s in timing_sizes]

    print("=== Проверка перемножения матриц ===")
    log_path = os.path.join("verification.log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        for size in manual_sizes:
            verify_multiplication(size, log_file)

    print(f"\nРезультаты проверки записаны в файл: {log_path}")

    print("\n=== Построение графика времени ===")
    plot_timing(timing_sizes, times)


if __name__ == "__main__":
    main()
