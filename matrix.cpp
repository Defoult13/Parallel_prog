#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <omp.h>

using namespace std;
using namespace std::chrono;

void generateMatrix(const string& filename, int rows, int cols) {
    ofstream file(filename);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1000000.0, 1000000.0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double val = dis(gen);
            file << fixed << setprecision(2) << val << " ";
        }
        file << "\n";
    }

    file.close();
}

vector<vector<double>> readMatrix(const string& filename, int rows, int cols) {
    ifstream file(filename);
    vector<vector<double>> matrix(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            file >> matrix[i][j];
    file.close();
    return matrix;
}

vector<vector<double>> multiplyMatrices(const vector<vector<double>>& A, const vector<vector<double>>& B, int n, int m, int p) {
    vector<vector<double>> result(n, vector<double>(p, 0.0));

    #pragma omp parallel num_threads(8)
    {
        #pragma omp for
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < p; ++j) {
                double sum = 0.0;

                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < m; ++k) {
                    sum += A[i][k] * B[k][j];
                }

                result[i][j] = sum;
            }
        }
    }

    return result;
}

void writeMatrix(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);
    for (const auto& row : matrix) {
        for (double val : row) {
            file << fixed << setprecision(2) << val << " ";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    setlocale(LC_ALL, "");

    const int numPairs = 10;

    ofstream timingFile("timing_results.txt");
    timingFile << "Размерность\tСреднее время (сек)\n";

    for (int size = 50; size <= 1000; size += 50) {
        int n = size, m = size, p = size;
        double totalTime = 0.0;

        cout <<"Обработка размерности: " << size << "x" << size << "...\n";

        for (int i = 1; i <= numPairs; ++i) {
            string fileA = "MatrixA(" + to_string(size) + ")_" + to_string(i) + ".txt";
            string fileB = "MatrixB(" + to_string(size) + ")_" + to_string(i) + ".txt";
            string fileResult = "Result(" + to_string(size) + ")_" + to_string(i) + ".txt";

            generateMatrix(fileA, n, m);
            generateMatrix(fileB, m, p);

            auto A = readMatrix(fileA, n, m);
            auto B = readMatrix(fileB, m, p);

            auto start = high_resolution_clock::now();
            auto result = multiplyMatrices(A, B, n, m, p);
            auto end = high_resolution_clock::now();
            totalTime += duration<double>(end - start).count();

            writeMatrix(fileResult, result);
        }

        double averageTime = totalTime / numPairs;
        timingFile << size << "x" << size << "\t" << fixed << setprecision(6) << averageTime << "\n";
    }

    timingFile.close();
    cout << "\nГотово! Все матрицы и результаты сохранены. Время выполнения записано в timing_results.txt\n";

    return 0;
}