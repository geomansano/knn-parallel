#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

typedef struct {
    float dist;
    int label;
} DistLabel;

// ---------- COUNT LINES ----------
int count_lines(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;
    
    int lines = 0;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), fp)) lines++;

    fclose(fp);
    return lines;
}

// ---------- SWAP ----------
void swap(DistLabel *a, DistLabel *b) {
    DistLabel t = *a;
    *a = *b;
    *b = t;
}

// ---------- PARTITION ----------
int partition(DistLabel arr[], int left, int right) {
    float pivot = arr[right].dist;
    int i = left;
    for (int j = left; j < right; j++) {
        if (arr[j].dist <= pivot) {
            swap(&arr[i], &arr[j]);
            i++;
        }
    }
    swap(&arr[i], &arr[right]);
    return i;
}

// ---------- QUICKSELECT ----------
void quickselect(DistLabel arr[], int left, int right, int k) {
    if (left < right) {
        int pivotIndex = partition(arr, left, right);
        if (pivotIndex == k) return;
        if (k < pivotIndex) quickselect(arr, left, pivotIndex - 1, k);
        else quickselect(arr, pivotIndex + 1, right, k);
    }
}

// =====================================================================
// -------------------------- CUDA KERNEL ------------------------------
// =====================================================================
__global__ void compute_distances(
    const float *d_X_train,
    const float *d_test_point,
    int n_train,
    int n_features,
    DistLabel *d_out_dist,
    const int *d_y_train
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_train) return;

    float sum = 0.0f;

    for (int f = 0; f < n_features; f++) {
        float diff = d_test_point[f] -
                     d_X_train[idx * n_features + f];
        sum += diff * diff;
    }

    d_out_dist[idx].dist = sqrtf(sum);
    d_out_dist[idx].label = d_y_train[idx];
}

// =====================================================================
// ----------------------- KNN CUDA SIMPLES ----------------------------
// =====================================================================
int knn_predict_cuda(
    float *h_X_train, int *h_y_train,
    int n_train,
    float *h_test_point,
    int n_features,
    int k
) {
    float *d_X_train, *d_test_point;
    int *d_y_train;
    DistLabel *d_dist_list;

    cudaMalloc(&d_X_train, n_train * n_features * sizeof(float));
    cudaMalloc(&d_y_train, n_train * sizeof(int));
    cudaMalloc(&d_test_point, n_features * sizeof(float));
    cudaMalloc(&d_dist_list, n_train * sizeof(DistLabel));

    cudaMemcpy(d_X_train, h_X_train,
               n_train * n_features * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_y_train, h_y_train,
               n_train * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_test_point, h_test_point,
               n_features * sizeof(float),
               cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n_train + blockSize - 1) / blockSize;

    compute_distances<<<gridSize, blockSize>>>(
        d_X_train,
        d_test_point,
        n_train,
        n_features,
        d_dist_list,
        d_y_train
    );

    cudaDeviceSynchronize();

    DistLabel *h_dist_list =
        (DistLabel*) malloc(n_train * sizeof(DistLabel));

    cudaMemcpy(h_dist_list, d_dist_list,
               n_train * sizeof(DistLabel),
               cudaMemcpyDeviceToHost);

    // CPU quickselect
    quickselect(h_dist_list, 0, n_train - 1, k);

    int c0 = 0, c1 = 0;
    for (int i = 0; i < k; i++) {
        if (h_dist_list[i].label == 0) c0++;
        else c1++;
    }

    int result = (c1 > c0 ? 1 : 0);

    free(h_dist_list);
    cudaFree(d_X_train);
    cudaFree(d_y_train);
    cudaFree(d_test_point);
    cudaFree(d_dist_list);

    return result;
}

// =====================================================================
// ------------------------------ MAIN --------------------------------
// =====================================================================
int main() {

    const char *train_file = "dados_treino.csv";
    const char *test_file  = "dados_teste.csv";

    int n_train = count_lines(train_file) - 1;
    int n_test  = count_lines(test_file) - 1;

    int num_features = 36;

    float *X_train = (float*) malloc(n_train * num_features * sizeof(float));
    float *X_test  = (float*) malloc(n_test  * num_features * sizeof(float));

    int *y_train = (int*) malloc(n_train * sizeof(int));
    int *y_test  = (int*) malloc(n_test  * sizeof(int));

    FILE *fp_train = fopen(train_file, "r");
    FILE *fp_test  = fopen(test_file, "r");

    char line[1024];
    fgets(line, sizeof(line), fp_train);
    fgets(line, sizeof(line), fp_test);

    int row = 0;
    while (fgets(line, sizeof(line), fp_train) && row < n_train) {
        char *token = strtok(line, ",");
        for (int col = 0; col < num_features; col++) {
            X_train[row * num_features + col] = atof(token);
            token = strtok(NULL, ",");
        }
        y_train[row] = atoi(token);
        row++;
    }

    row = 0;
    while (fgets(line, sizeof(line), fp_test) && row < n_test) {
        char *token = strtok(line, ",");
        for (int col = 0; col < num_features; col++) {
            X_test[row * num_features + col] = atof(token);
            token = strtok(NULL, ",");
        }
        y_test[row] = atoi(token);
        row++;
    }

    // ======================= KNN CUDA (SIMPLIFICADO) ======================= 

    int *h_pred = (int*) malloc(n_test * sizeof(int));
    int k = 5;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("\nRodando KNN CUDA simples...\n");

    cudaEventRecord(start);

    for (int i = 0; i < n_test; i++) { // 
        h_pred[i] = knn_predict_cuda(
            X_train,
            y_train,
            n_train,
            &X_test[i * num_features],
            num_features,
            k
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Tempo total CUDA: %.4f ms\n", ms);

    // Acurácia
    int correct = 0;
    for (int i = 0; i < n_test; i++)
        if (h_pred[i] == y_test[i]) correct++;

    float acc = (float) correct / n_test * 100.0f;
    printf("Acurácia: %.2f%%\n", acc);

    // Liberando Memória
    free(X_train);
    free(X_test);
    free(y_train);
    free(y_test);
    free(h_pred);

    return 0;
}
