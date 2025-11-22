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
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Número de execuções do algoritmo
#define N_RUNS 5

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

// Função KNN
int knn_predict(
    float **X_train, int *y_train, int n_train,
    float *test_point, int n_features, int k
) {
    // Cria vetor com distâncias
    DistLabel *dist_list = (DistLabel *) malloc(n_train * sizeof(DistLabel));
	
	// Calcula a distância euclididana do ponto de teste para todos os pontos de treino
    for(int i = 0; i < n_train; i++) {
        dist_list[i].dist = euclidean_distance(test_point, X_train[i], n_features);
        dist_list[i].label = y_train[i];
    }
	
	// Printando os 15 primeiros elementos do vetor antes da ordenação
	// for(int i = 0; i < 15; i++)
    // printf("[%2d] dist = %.4f | label = %d\n", i, dist_list[i].dist, dist_list[i].label);
	
    // QuickSelect para achar k menores distâncias
    quickselect(dist_list, 0, n_train - 1, k);
    
    // Printando os 15 primeiros elementos do vetor após a ordenação
	// for(int i = 0; i < 15; i++)
    // printf("[%2d] dist = %.4f | label = %d\n", i, dist_list[i].dist, dist_list[i].label);

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
	// Arquivos de treino e teste
	const char *train_file = "dados_treino.csv";
    const char *test_file = "dados_teste.csv";
    
	// Contando as linhas e descontando o cabeçalho
	int n_train = count_lines(train_file) - 1;
    int n_test = count_lines(test_file) - 1; 
    
    // Verificando se a contagem foi feita corretamente
    if (n_train <= 0 || n_test <= 0) {
        printf("Erro ao contar linhas do arquivo\n");
        return 1;
    }
    
    // Exibindo tamanho dos datasets
    printf("Treino: %d linhas, Teste: %d linhas\n", n_train, n_test);
	
	// Número de features
    int num_features = 36; // número de colunas menos a coluna target

    // Alocando memória para os dados de treino	
    float **X_train = (float **)malloc(n_train * sizeof(float *));
    for (int i = 0; i < n_train; i++)
        X_train[i] = (float *)malloc(num_features * sizeof(float));
    int *y_train = (int *)malloc(n_train * sizeof(int));

    // Alocando memória para os dados de teste
    float **X_test = (float **)malloc(n_test * sizeof(float *));
    for (int i = 0; i < n_test; i++)
        X_test[i] = (float *)malloc(num_features * sizeof(float));
    int *y_test = (int *)malloc(n_test * sizeof(int));

    // Lendo os CSVs
    FILE *fp_train = fopen(train_file, "r");
    FILE *fp_test = fopen(test_file, "r");
    if (!fp_train || !fp_test) {
        printf("Erro ao abrir arquivos CSV\n");
        return 1;
    }

    char line[1024];
    int row = 0;
    
    // Pula o cabeçalho
    if (fgets(line, sizeof(line), fp_train) == NULL) { }
    if (fgets(line, sizeof(line), fp_test) == NULL) { }

    // CSV de treino
    while (fgets(line, sizeof(line), fp_train) && row < n_train) {
        char *token = strtok(line, ",");
        for (int col = 0; col < num_features; col++) {
            X_train[row][col] = atof(token);
            token = strtok(NULL, ",");
        }
        y_train[row] = atoi(token);
        row++;
    }

    // CSV de teste
    row = 0;
    while (fgets(line, sizeof(line), fp_test) && row < n_test) {
        char *token = strtok(line, ",");
        for (int col = 0; col < num_features; col++) {
            X_test[row][col] = atof(token);
            token = strtok(NULL, ",");
        }
        y_test[row] = atoi(token);
        row++;
    }

    fclose(fp_train);
    fclose(fp_test);

    printf("CSV carregado dinamicamente!\n");
    
    // Quantidade máxima de elementos a imprimir
	int max_print = 10;

	// Imprimindo os primeiros 10 elementos de X_train e y_train
	printf("\n--- Primeiros elementos de X_train e y_train ---\n");
	for (int i = 0; i < max_print; i++) {
		printf("Linha %d: ", i);
		for (int j = 0; j < num_features; j++) {
			printf("%.2f ", X_train[i][j]);
		}
		printf("| Target: %d\n", y_train[i]);
	}

	// Imprimindo os primeiros 10 elementos de X_test e y_test
	printf("\n--- Primeiros elementos de X_test e y_test ---\n");
	for (int i = 0; i < max_print; i++) {
		printf("Linha %d: ", i);
		for (int j = 0; j < num_features; j++) {
			printf("%.2f ", X_test[i][j]);
		}
		printf("| Target: %d\n", y_test[i]);
	}
	
	// ======================= APLICANDO O KNN SEQUENCIAL =======================
	
	// Valor do K
	int k = 5;

	// Vetor de predições
	int *y_predict = (int *) malloc(n_test * sizeof(int));
	
	double times_seq[N_RUNS]; // armazenar tempos individuais

	printf("\nAplicando KNN sequencial (%d execuções)...\n", N_RUNS);

	for(int run = 0; run < N_RUNS; run++)
	{
		struct timespec start, end;
		clock_gettime(CLOCK_MONOTONIC, &start);

		for (int i = 0; i < n_test; i++) {
			y_predict[i] = knn_predict(X_train, y_train, n_train, X_test[i], num_features, k);

			if(i % 500 == 0)
				printf("[Run %d] Processando amostra %d de %d...\n", run+1, i, n_test);
		}

		clock_gettime(CLOCK_MONOTONIC, &end);
		double elapsed = (end.tv_sec - start.tv_sec) +
						 (end.tv_nsec - start.tv_nsec) / 1e9;

		times_seq[run] = elapsed;
		printf("\n--> Tempo da execução %d: %.6f segundos\n", run+1, elapsed);
	}

	// Cálculo da média
	double avg = 0.0;
	for(int i = 0; i < N_RUNS; i++)
		avg += times_seq[i];
	avg /= N_RUNS;

	printf(" Tempos individuais (s):\n");
	for(int i = 0; i < N_RUNS; i++)
		printf("  Execução %d: %.6f s\n", i+1, times_seq[i]);

	printf("\n Média final do KNN sequencial: %.6f segundos\n", avg);
	
	// Cálculo da acurácia da última iteração
	int correct = 0;
	for (int i = 0; i < n_test; i++) {
		if (y_predict[i] == y_test[i]) correct++;
	}
	
	float acc = (float) correct / n_test * 100.0f;
	printf("\nAcurácia última iteração KNN sequencial (k=%d): %.2f%%\n", k, acc);
	
    // Liberar memória
    for (int i = 0; i < n_train; i++) free(X_train[i]);
    free(X_train); free(y_train);
    for (int i = 0; i < n_test; i++) free(X_test[i]);
    free(X_test); free(y_test);
    free(y_predict);
    
	return 0;
}
	
