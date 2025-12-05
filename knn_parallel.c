#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Estrutura para armazenar a distância ponto a ponto e o rótulo
typedef struct {
    float dist;
    int label;
} DistLabel;

// Conta as linhas do CSV
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

// Cálculo da Distância Euclidiana 
float euclidean_distance(const float *train_row, const float *test_row, int n_features) {
    float sum = 0.0f;
    for (int i = 0; i < n_features; i++) {
        float diff = train_row[i] - test_row[i];
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
    const float *X_train,
    const int *y_train,
    int n_train,
    const float *test_point,
    int n_features,
    int k
) {
    DistLabel *dist_list = malloc(n_train * sizeof(DistLabel));

    for (int i = 0; i < n_train; i++) {
        const float *train_row = &X_train[i * n_features];
        dist_list[i].dist = euclidean_distance(train_row, test_point, n_features);
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
	
	// Contador do label mais comum 
    int c0 = 0, c1 = 0;
    for (int i = 0; i < k; i++) {
        if (dist_list[i].label == 0) c0++;
        else c1++;
    }
	
    free(dist_list);

    return (c1 > c0);
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
	
	// Alocação linear
	float *X_train = (float*) malloc(n_train * num_features * sizeof(float));
    float *X_test  = (float*) malloc(n_test  * num_features * sizeof(float));

    int *y_train = (int*) malloc(n_train * sizeof(int));
    int *y_test  = (int*) malloc(n_test  * sizeof(int));
    
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

    // CSV Treino
    while (fgets(line, sizeof(line), fp_train) && row < n_train) {
        char *token = strtok(line, ",");

        for (int col = 0; col < num_features; col++) {
            X_train[row * num_features + col] = atof(token);
            token = strtok(NULL, ",");
        }

        y_train[row] = atoi(token);
        row++;
    }

    // CSV Teste
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
    
    fclose(fp_train);
    fclose(fp_test);
    
    /*
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
	*/
	
	// ======================= KNN SEQUENCIAL =======================
	int *y_pred = malloc(n_test * sizeof(int));
    int k = 5;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

	//printf("\nAplicando KNN sequencial...\n");
    for (int i = 0; i < n_test; i++) { // n_test
        y_pred[i] = knn_predict(
            X_train, y_train, n_train,
            &X_test[i * num_features],
            num_features, k
        );
        //if(i % 500 == 0)  // feedback a cada 500 linhas
		//	printf("Processando amostra %d de %d...\n", i, n_test);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec)
                    + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("\nTempo sequencial: %.6f s\n", elapsed);

    int correct = 0;
    for (int i = 0; i < n_test; i++)
        if (y_pred[i] == y_test[i]) correct++;

    printf("Acurácia: %.2f%%\n",
        100.0f * correct / n_test
    );

    // Libera memória
    free(X_train);
    free(X_test);
    free(y_train);
    free(y_test);
    free(y_pred);
    
	return 0;
}
