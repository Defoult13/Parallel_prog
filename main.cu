#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <cuda_runtime.h>

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

__global__ void matrixMultiplyKernel(const double* A, const double* B, double* C, int n, int m, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < p) {
        double sum = 0.0;
        for (int k = 0; k < m; ++k) {
            sum += A[row * m + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

vector<vector<double>> gpuMultiply(const vector<vector<double>>& A, const vector<vector<double>>& B, int n, int m, int p) {
    vector<vector<double>> C(n, vector<double>(p, 0.0));

    vector<double> flatA(n * m), flatB(m * p), flatC(n * p);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            flatA[i * m + j] = A[i][j];

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < p; ++j)
            flatB[i * p + j] = B[i][j];

    double* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, flatA.size() * sizeof(double));
    cudaMalloc((void**)&d_B, flatB.size() * sizeof(double));
    cudaMalloc((void**)&d_C, flatC.size() * sizeof(double));

    cudaMemcpy(d_A, flatA.data(), flatA.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB.data(), flatB.size() * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks((p + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiplyKernel << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, n, m, p);
    cudaDeviceSynchronize();

    cudaMemcpy(flatC.data(), d_C, flatC.size() * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            C[i][j] = flatC[i * p + j];

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

int main() {
    setlocale(LC_ALL, "");

    const int numPairs = 10;

    ofstream timingFile("timing_results(8).txt");
    timingFile << "Размерность\tСреднее время (сек)\n";

    for (int size = 50; size <= 1000; size += 50) {
        int n = size, m = size, p = size;
        double totalTime = 0.0;

        cout << "Обработка размерности: " << size << "x" << size << "...\n";

        for (int i = 1; i <= numPairs; ++i) {
            string fileA = "MatrixA(" + to_string(size) + ")_" + to_string(i) + ".txt";
            string fileB = "MatrixB(" + to_string(size) + ")_" + to_string(i) + ".txt";
            string fileResult = "Result(" + to_string(size) + ")_" + to_string(i) + ".txt";

            generateMatrix(fileA, n, m);
            generateMatrix(fileB, m, p);

            auto A = readMatrix(fileA, n, m);
            auto B = readMatrix(fileB, m, p);

            auto start = high_resolution_clock::now();
            auto result = gpuMultiply(A, B, n, m, p);
            auto end = high_resolution_clock::now();
            totalTime += duration<double>(end - start).count();

            writeMatrix(fileResult, result);
        }

        double averageTime = totalTime / numPairs;
        timingFile << size << "x" << size << "\t" << fixed << setprecision(6) << averageTime << "\n";
    }

    timingFile.close();
    cout << "\nГотово! Все матрицы и результаты сохранены. Время выполнения записано в timing_results(8).txt\n";

    return 0;
}
