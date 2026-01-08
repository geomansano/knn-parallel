// knn_cuda_batch.cu
// Versão C: KNN full GPU por batches (batch_size = 1024)
// 3 kernels: (1) distâncias, (2) quickselect por linha, (3) votação
// Compilar: nvcc -O2 knn_cuda_batch.cu -o knn_cuda_batch

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

#define CHECK_CUDA(call) do {                                 \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1);                                              \
    }                                                         \
} while(0)

// ------------------------------------------------------------------
// Host helpers
// ------------------------------------------------------------------
int count_lines(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;
    int lines = 0;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), fp)) lines++;
    fclose(fp);
    return lines;
}

// ------------------------------------------------------------------
// Device helpers for quickselect (operate on DistLabel array)
// ------------------------------------------------------------------
__device__ inline void swap_dl(DistLabel *a, DistLabel *b) {
    DistLabel tmp = *a;
    *a = *b;
    *b = tmp;
}

__device__ int partition_device(DistLabel *arr, int left, int right) {
    float pivot = arr[right].dist;
    int i = left;
    for (int j = left; j < right; ++j) {
        if (arr[j].dist <= pivot) {
            swap_dl(&arr[i], &arr[j]);
            ++i;
        }
    }
    swap_dl(&arr[i], &arr[right]);
    return i;
}

// Iterative quickselect (device) to partition such that first k are the k smallest (unordered)
__device__ void quickselect_device(DistLabel *arr, int left, int right, int k) {
    while (left < right) {
        int p = partition_device(arr, left, right);
        if (p == k) return;
        else if (k < p) right = p - 1;
        else left = p + 1;
    }
}

// ------------------------------------------------------------------
// Kernel 1: compute distances for a batch of test points
// Each thread computes one (train_id, test_id) pair
// Grid is 2D: x -> train_id, y -> test_in_batch
// ------------------------------------------------------------------
__global__ void compute_distances_batch(
    const float *d_X_train,   // size n_train * n_features
    const float *d_X_test_b,  // size batch_cur * n_features
    const int   *d_y_train,   // size n_train
    DistLabel   *d_distlab,   // size batch_cur * n_train
    int n_train,
    int batch_cur,
    int n_features
) {
    int train_id = blockIdx.x * blockDim.x + threadIdx.x;
    int test_id  = blockIdx.y * blockDim.y + threadIdx.y;

    if (train_id >= n_train || test_id >= batch_cur) return;

    // compute linear offsets
    const float *train_ptr = d_X_train + (size_t)train_id * n_features;
    const float *test_ptr  = d_X_test_b  + (size_t)test_id  * n_features;

    float sum = 0.0f;
    for (int f = 0; f < n_features; ++f) {
        float diff = test_ptr[f] - train_ptr[f];
        sum += diff * diff;
    }
    float dist = sqrtf(sum);

    // store into distlab at index test_id * n_train + train_id
    size_t idx = (size_t)test_id * n_train + train_id;
    d_distlab[idx].dist  = dist;
    d_distlab[idx].label = d_y_train[train_id];
}

// ------------------------------------------------------------------
// Kernel 2: quickselect per test (one thread per test in batch)
// Each thread calls quickselect_device on its row: d_distlab[test * n_train ..]
// ------------------------------------------------------------------
__global__ void quickselect_per_test(
    DistLabel *d_distlab,
    int n_train,
    int batch_cur,
    int k
) {
    int test_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (test_id >= batch_cur) return;

    DistLabel *row = d_distlab + (size_t)test_id * n_train;
    // quickselect to partition first k elements
    quickselect_device(row, 0, n_train - 1, k - 1); // partition so indices 0..k-1 are k smallest
}

// ------------------------------------------------------------------
// Kernel 3: vote per test (one thread per test in batch)
// After quickselect, first k of each row are k nearest (unordered).
// ------------------------------------------------------------------
__global__ void vote_knn_batch(
    DistLabel *d_distlab,
    int batch_cur,
    int n_train,
    int k,
    int *d_pred_out
) {
    int test_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (test_id >= batch_cur) return;

    DistLabel *row = d_distlab + (size_t)test_id * n_train;
    int count0 = 0, count1 = 0;
    for (int i = 0; i < k; ++i) {
        int lab = row[i].label;
        if (lab == 0) ++count0;
        else ++count1;
    }
    d_pred_out[test_id] = (count1 > count0) ? 1 : 0;
}

// ------------------------------------------------------------------
// Host: main
// ------------------------------------------------------------------
int main(int argc, char **argv) {
    const char *train_file = "dados_treino.csv";
    const char *test_file  = "dados_teste.csv";

    // parameters
    const int num_features = 36;
    const int k = 5;
    const int batch_size = 3072; 

    // read counts
    int n_train = count_lines(train_file) - 1;
    int n_test  = count_lines(test_file) - 1;
    if (n_train <= 0 || n_test <= 0) {
        fprintf(stderr, "Erro contando linhas (treino=%d, teste=%d)\n", n_train, n_test);
        return 1;
    }
    printf("Treino: %d linhas, Teste: %d linhas\n", n_train, n_test);

    // allocate host linear arrays
    float *h_X_train = (float*) malloc((size_t)n_train * num_features * sizeof(float));
    float *h_X_test  = (float*) malloc((size_t)n_test  * num_features * sizeof(float));
    int   *h_y_train = (int*)   malloc((size_t)n_train * sizeof(int));
    int   *h_y_test  = (int*)   malloc((size_t)n_test  * sizeof(int));
    if (!h_X_train || !h_X_test || !h_y_train || !h_y_test) {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    // read CSVs (assumes no missing values and last column is label)
    FILE *fp_train = fopen(train_file, "r");
    FILE *fp_test  = fopen(test_file,  "r");
    if (!fp_train || !fp_test) {
        fprintf(stderr, "Erro abrindo arquivos CSV\n");
        return 1;
    }

    char line[16384];
    // skip header
    if (fgets(line, sizeof(line), fp_train) == NULL) { }
    if (fgets(line, sizeof(line), fp_test) == NULL) { }
	
	int row = 0;
    while (fgets(line, sizeof(line), fp_train) && row < n_train) {
        char *token = strtok(line, ",");
        for (int col = 0; col < num_features; col++) {
            h_X_train[(size_t)row * num_features + col] = atof(token);
            token = strtok(NULL, ",");
        }
        h_y_train[row] = atoi(token);
        row++;
    }

    row = 0;
    while (fgets(line, sizeof(line), fp_test) && row < n_test) {
        char *token = strtok(line, ",");
        for (int col = 0; col < num_features; col++) {
            h_X_test[(size_t)row * num_features + col] = atof(token);
            token = strtok(NULL, ",");
        }
        h_y_test[row] = atoi(token);
        row++;
    }

    fclose(fp_train);
    fclose(fp_test);

    // copy train data to device once
    float *d_X_train = NULL;
    int   *d_y_train = NULL;
    CHECK_CUDA(cudaMalloc(&d_X_train, (size_t)n_train * num_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y_train, (size_t)n_train * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_X_train, h_X_train, (size_t)n_train * num_features * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_train, h_y_train, (size_t)n_train * sizeof(int), cudaMemcpyHostToDevice));

    // allocate buffers used per-batch on device
    // max batch = batch_size * n_train DistLabel structs
    size_t max_batch = (size_t)batch_size;
    DistLabel *d_distlab = NULL;
    CHECK_CUDA(cudaMalloc(&d_distlab, max_batch * (size_t)n_train * sizeof(DistLabel)));

    // device buffer for test batch features
    float *d_X_test_b = NULL;
    CHECK_CUDA(cudaMalloc(&d_X_test_b, (size_t)batch_size * num_features * sizeof(float)));

    // device buffer for predictions (per batch)
    int *d_pred_batch = NULL;
    CHECK_CUDA(cudaMalloc(&d_pred_batch, (size_t)batch_size * sizeof(int)));

    // host buffer for predictions final
    int *h_pred = (int*) malloc((size_t)n_test * sizeof(int));
    if (!h_pred) { fprintf(stderr, "Alloc h_pred failed\n"); return 1; }

    // configure kernel launch parameters
    dim3 block_dist (256, 4); // tuneable: x: train-tile, y: test-tile (32, 8)
    // compute grid for kernel1 dynamically per batch: 
    // grid.x = ceil(n_train/block_dist.x)
    // grid.y = ceil(batch_cur / block_dist.y)

    // quickselect kernel: one thread per test in batch
    int block_qs = 128; // 384
    // voting kernel uses same config

    // Timing events (measure whole GPU KNN time)
    cudaEvent_t start_evt, stop_evt;
    CHECK_CUDA(cudaEventCreate(&start_evt));
    CHECK_CUDA(cudaEventCreate(&stop_evt));
    CHECK_CUDA(cudaEventRecord(start_evt));

    // process test set in batches
    for (int offset = 0; offset < n_test; offset += batch_size) {
        int batch_cur = ((offset + batch_size) <= n_test) ? batch_size : (n_test - offset);

        // copy batch features
        CHECK_CUDA(cudaMemcpy(d_X_test_b, &h_X_test[(size_t)offset * num_features],
                              (size_t)batch_cur * num_features * sizeof(float),
                              cudaMemcpyHostToDevice));

        // launch kernel1: compute distances for this batch
        dim3 grid_dist( (n_train + block_dist.x - 1) / block_dist.x,
                        (batch_cur + block_dist.y - 1) / block_dist.y );

        compute_distances_batch<<<grid_dist, block_dist>>>(
            d_X_train, d_X_test_b, d_y_train, d_distlab, n_train, batch_cur, num_features
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // kernel2: quickselect per test (one thread per test in batch)
        int grid_qs = (batch_cur + block_qs - 1) / block_qs;
        quickselect_per_test<<<grid_qs, block_qs>>>(d_distlab, n_train, batch_cur, k);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // kernel3: vote (one thread per test in batch)
        vote_knn_batch<<<grid_qs, block_qs>>>(d_distlab, batch_cur, n_train, k, d_pred_batch);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // copy predictions back to host for this batch
        CHECK_CUDA(cudaMemcpy(&h_pred[offset], d_pred_batch, (size_t)batch_cur * sizeof(int), cudaMemcpyDeviceToHost));
    }

    // stop timing
    CHECK_CUDA(cudaEventRecord(stop_evt));
    CHECK_CUDA(cudaEventSynchronize(stop_evt));
    float ms_total = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_total, start_evt, stop_evt));
    printf("Tempo total GPU (todos os batches): %.3f ms\n", ms_total);

    // compute accuracy (if you have h_y_test)
    int correct = 0;
    for (int i = 0; i < n_test; ++i) if (h_pred[i] == h_y_test[i]) ++correct;
    float acc = (float)correct / n_test * 100.0f;
    printf("Acurácia: %.2f %% (%d/%d)\n", acc, correct, n_test);

    // free
    free(h_X_train);
    free(h_X_test);
    free(h_y_train);
    free(h_y_test);
    free(h_pred);

    CHECK_CUDA(cudaFree(d_X_train));
    CHECK_CUDA(cudaFree(d_y_train));
    CHECK_CUDA(cudaFree(d_distlab));
    CHECK_CUDA(cudaFree(d_X_test_b));
    CHECK_CUDA(cudaFree(d_pred_batch));

    CHECK_CUDA(cudaEventDestroy(start_evt));
    CHECK_CUDA(cudaEventDestroy(stop_evt));

    return 0;
}
