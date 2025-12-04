/*
 * knn_parallel.c
 *
 * Copyright 2025 Geovanne <geovanne@geovanne-H610M-HVS-M-2-R2-0>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

// Estrutura para armazenar a distância ponto a ponto e o rótulo
typedef struct {
    float dist;
    int label;
} DistLabel;

// Conta a quantidade de linhas dos datasets para leitura
int count_lines(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;

    int lines = 0;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), fp)) {
        lines++;
    }
    fclose(fp);
    return lines;
}

// Cálculo da distância Euclidiana
float euclidean_distance(const float *a, const float *b, int n_features) {
    float sum = 0.0f;
    for(int i = 0; i < n_features; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

// Troca o conteúdo entre duas estruturas DistLabel (distância + label).
void swap(DistLabel *a, DistLabel *b) {
    DistLabel t = *a;
    *a = *b;
    *b = t;
}

// Particiona o array pelo pivô, colocando menores distâncias à esquerda e maiores à direita.
int partition(DistLabel arr[], int left, int right) {
    float pivot = arr[right].dist;
    int i = left;
    for(int j = left; j < right; j++) {
        if(arr[j].dist <= pivot) {
            swap(&arr[i], &arr[j]);
            i++;
        }
    }
    swap(&arr[i], &arr[right]);
    return i;
}

// Algoritmo de ordenação Quick Selection
void quickselect(DistLabel arr[], int left, int right, int k) {
    if(left < right) {
        int pivotIndex = partition(arr, left, right);
        if(pivotIndex == k) return;
        if(k < pivotIndex)
            quickselect(arr, left, pivotIndex - 1, k);
        else
            quickselect(arr, pivotIndex + 1, right, k);
    }
}

// Função KNN — espera X_train como array de ponteiros (float **)
int knn_predict(
    float **X_train, int *y_train, int n_train,
    float *test_point, int n_features, int k
) {
    // Cria vetor com distâncias
    DistLabel *dist_list = (DistLabel *) malloc(n_train * sizeof(DistLabel));
    if (!dist_list) {
        fprintf(stderr, "knn_predict: malloc falhou\n");
        return 0;
    }

    // Calcula a distância euclididana do ponto de teste para todos os pontos de treino
    for(int i = 0; i < n_train; i++) {
        dist_list[i].dist = euclidean_distance(test_point, X_train[i], n_features);
        dist_list[i].label = y_train[i];
    }

    // QuickSelect para achar k menores distâncias
    quickselect(dist_list, 0, n_train - 1, k);

    // Contador do label mais comum (apenas 0 e 1)
    int count0 = 0, count1 = 0;
    for(int i = 0; i < k; i++) {
        if(dist_list[i].label == 0) count0++;
        else count1++;
    }

    free(dist_list);

    return (count1 > count0) ? 1 : 0;
}

int main(int argc, char **argv)
{
    int *y_test = NULL, *y_train = NULL;
    float *X_test = NULL, *X_train = NULL; /* buffers linearizados */
    int inicializacao, num_features = 0, num_procs = 0, n_test = 0, n_train = 0, rank = 0;

    int base = 0, rem = 0, local_n_test = 0, start_idx = 0;
    float *X_test_local = NULL;
    int *y_test_local = NULL;

    inicializacao = MPI_Init(&argc, &argv);
    if (inicializacao != MPI_SUCCESS) {
        fprintf(stderr, "Erro ao iniciar o ambiente MPI - código de erro: %d\n", inicializacao);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (rank == 0) {
        printf("rank: %d\n", rank);
        printf("num_procs: %d\n", num_procs);
    }

    if (rank == 0) {
        // Arquivos de treino e teste
        const char *train_file = "dados_treino.csv";
        const char *test_file = "dados_teste.csv";

        // Contando as linhas e descontando o cabeçalho
        n_train = count_lines(train_file) - 1;
        n_test = count_lines(test_file) - 1;

        // Verificando se a contagem foi feita corretamente
        if (n_train <= 0 || n_test <= 0) {
            fprintf(stderr, "Erro ao contar linhas do arquivo\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Exibindo tamanho dos datasets
        printf("Treino: %d linhas, Teste: %d linhas\n", n_train, n_test);

        // Número de features
        num_features = 36; // número de colunas menos a coluna target

        // calcular divisão de teste
        base = n_test / num_procs;
        rem  = n_test % num_procs;

        // Alocando buffers linearizados (uma única região contínua)
        X_train = (float *)malloc((size_t)n_train * num_features * sizeof(float));
        y_train = (int *)malloc((size_t)n_train * sizeof(int));
        X_test  = (float *)malloc((size_t)n_test  * num_features * sizeof(float));
        y_test  = (int *)malloc((size_t)n_test  * sizeof(int));
        if (!X_train || !y_train || !X_test || !y_test) {
            fprintf(stderr, "Erro de alocação de memória no root\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Lendo os CSVs
        FILE *fp_train = fopen(train_file, "r");
        FILE *fp_test = fopen(test_file, "r");
        if (!fp_train || !fp_test) {
            fprintf(stderr, "Erro ao abrir arquivos CSV\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char line[4096];
        int row = 0;

        // Pula o cabeçalho
        if (fgets(line, sizeof(line), fp_train) == NULL) { }
        if (fgets(line, sizeof(line), fp_test) == NULL) { }

        // CSV de treino
        row = 0;
        while (fgets(line, sizeof(line), fp_train) && row < n_train) {
            char *token = strtok(line, ",");
            for (int col = 0; col < num_features; col++) {
                if (!token) token = "0";
                X_train[row * num_features + col] = atof(token);
                token = strtok(NULL, ",");
            }
            if (!token) token = "0";
            y_train[row] = atoi(token);
            row++;
        }

        // CSV de teste
        row = 0;
        while (fgets(line, sizeof(line), fp_test) && row < n_test) {
            char *token = strtok(line, ",");
            for (int col = 0; col < num_features; col++) {
                if (!token) token = "0";
                X_test[row * num_features + col] = atof(token);
                token = strtok(NULL, ",");
            }
            if (!token) token = "0";
            y_test[row] = atoi(token);
            row++;
        }

        fclose(fp_train);
        fclose(fp_test);

        // Enviar metadados e X_train/y_train para escravos (1..num_procs-1)
        for (int i = 1; i < num_procs; i++) {
            MPI_Send(&num_features, 1, MPI_INT, i, 0, MPI_COMM_WORLD);                  // tag 0: metadata
            MPI_Send(&n_train, 1, MPI_INT, i, 1, MPI_COMM_WORLD);                       // tag 1: n_train
            MPI_Send(X_train, n_train * num_features, MPI_FLOAT, i, 2, MPI_COMM_WORLD); // tag 2: X_train
            MPI_Send(y_train, n_train, MPI_INT, i, 3, MPI_COMM_WORLD);                  // tag 3: y_train
        }

        // Distribuir pedaços de X_test/y_test
        for (int r = 0; r < num_procs; r++) {
            int r_local = (r < rem) ? (base + 1) : base;
            int r_start = (r < rem) ? (r * (base + 1)) : (rem * (base + 1) + (r - rem) * base);

            if (r == 0) {
                // o mestre não envia para si mesmo, apenas aponta para o sub-bloco
                local_n_test = r_local;
                start_idx = r_start;
                X_test_local = X_test + (size_t)start_idx * num_features;
                y_test_local = y_test + start_idx;
            } else {
                MPI_Send(&r_local, 1, MPI_INT, r, 10, MPI_COMM_WORLD);                                                       // tag 10 = local size
                MPI_Send(X_test + (size_t)r_start * num_features, r_local * num_features, MPI_FLOAT, r, 11, MPI_COMM_WORLD); // tag 11 = X_test chunk
                MPI_Send(y_test + r_start, r_local, MPI_INT, r, 12, MPI_COMM_WORLD);                                         // tag 12 = y_test chunk
            }
        }
    } else {
        // Receber metadados e X_train/y_train do root
        MPI_Recv(&num_features, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&n_train, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        X_train = (float *)malloc((size_t)n_train * num_features * sizeof(float));
        y_train = (int *)malloc((size_t)n_train * sizeof(int));
        if (!X_train || !y_train) {
            fprintf(stderr, "Erro de alocação de memória no rank %d\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_Recv(X_train, n_train * num_features, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(y_train, n_train, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Receber pedaço local de teste
        MPI_Recv(&local_n_test, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        X_test_local = (float *)malloc((size_t)local_n_test * num_features * sizeof(float));
        y_test_local = (int *)malloc((size_t)local_n_test * sizeof(int));
        if (!X_test_local || !y_test_local) {
            fprintf(stderr, "rank %d: falha alocar X_test_local/y_test_local (local_n_test=%d)\n", rank, local_n_test);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_Recv(X_test_local, local_n_test * num_features, MPI_FLOAT, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(y_test_local, local_n_test, MPI_INT, 0, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // calcular start_idx local (mesma lógica)
        base = n_test / num_procs;
        // Para garantir start_idx correto sem n_test, recomputamos com a mesma fórmula usando n_test total:
        // para isso, root poderia ter enviado n_test; é mais robuste enviar n_test também — vamos receber n_test no início.
        // Para simplicidade, vamos receber n_test no começo na próxima iteração. Aqui assumimos partição idêntica. 
        if (rank < 0) start_idx = 0; // placeholder — não usado se você não precisa do índice global local
    }

    printf("rank %d: CSV carregado dinamicamente! local_n_test=%d\n", rank, local_n_test);

    /* === preparar ponteiros para knn_predict: criar array de ponteiros para X_train ===
       knn_predict espera float ** (cada elemento X_train[i] apontando para i-th row)
    */
    float **train_rows = (float **)malloc((size_t)n_train * sizeof(float *));
    if (!train_rows) {
        fprintf(stderr, "rank %d: falha alocar train_rows\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < n_train; i++) {
        train_rows[i] = X_train + (size_t)i * num_features;
    }

    /* ======================= APLICANDO O KNN PARA O BLOCO LOCAL ======================= */

    int k = 5; // Valor do K (padrão)
    int *y_predict_local = (int *) malloc((size_t)local_n_test * sizeof(int));
    if (!y_predict_local) {
        fprintf(stderr, "rank %d: falha alocar y_predict_local\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Sincronizar e medir tempo com MPI_Wtime. O root imprimirá o tempo total */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Laço local: processar as amostras deste rank */
    for (int i = 0; i < local_n_test; i++) {
        float *test_point = X_test_local + (size_t)i * num_features;
        y_predict_local[i] = knn_predict(train_rows, y_train, n_train, test_point, num_features, k);
    }

    /* Recolher resultados no root usando Gatherv (cada rank envia local_n_test elementos) */
    int *counts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        counts = (int *)malloc((size_t)num_procs * sizeof(int));
        displs = (int *)malloc((size_t)num_procs * sizeof(int));
        if (!counts || !displs) {
            fprintf(stderr, "root: falha alocar counts/displs\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        /* recomputar counts/displs usando a mesma partição */
        for (int r = 0; r < num_procs; r++) {
            counts[r] = (r < rem) ? (base + 1) : base;
        }
        displs[0] = 0;
        for (int r = 1; r < num_procs; r++) displs[r] = displs[r-1] + counts[r-1];
    }

    int *y_predict_global = NULL;
    if (rank == 0) {
        y_predict_global = (int *)malloc((size_t)n_test * sizeof(int));
        if (!y_predict_global) {
            fprintf(stderr, "root: falha alocar y_predict_global\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gatherv(y_predict_local, local_n_test, MPI_INT,
                y_predict_global, counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        double elapsed = t1 - t0;
        printf("\nTempo paralelo (do barrier antes da classificação até recolher predições): %.6f segundos\n", elapsed);

        /* Calcular acurácia — root possui y_test global */
        int correct = 0;
        for (int i = 0; i < n_test; i++) {
            if (y_predict_global[i] == y_test[i]) correct++;
        }
        float acc = (float) correct / n_test * 100.0f;
        printf("\nAcurácia KNN paralela (k=%d): %.2f%% (%d/%d)\n", k, acc, correct, n_test);
    }

    /* ======================= Tratamento de memória e finalização ======================= */

    /* liberar recursos locais */
    free(train_rows);
    free(y_predict_local);

    if (rank == 0) {
        /* root: free global buffers */
        free(y_predict_global);
        free(counts);
        free(displs);

        /* root tinha X_test/y_test globais e X_test_local is a pointer into X_test; free global buffers */
        free(X_test);
        free(y_test);
        /* X_test_local points into X_test for root, so no separate free */
    } else {
        /* escravos: free locais de teste */
        free(X_test_local);
        free(y_test_local);
    }

    /* todos desalocam X_train/y_train e X_train foi alocado em cada rank */
    free(X_train);
    free(y_train);

    MPI_Finalize();

    return 0;
}
