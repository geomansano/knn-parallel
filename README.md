# Paralelização do KNN em C com CUDA, MPI e OpenMP

## Descrição do Projeto

Este repositório apresenta uma implementação completa e otimizada do algoritmo **K-Nearest Neighbors (KNN)** desenvolvida do zero em **linguagem C**, com foco em **computação de alto desempenho**.  
O trabalho explora diferentes modelos de paralelismo para acelerar as principais etapas do KNN, combinando:

- CUDA para paralelismo massivo em GPU  
- OpenMP para paralelismo em memória compartilhada na CPU  
- MPI para paralelismo distribuído entre múltiplos processos  

O projeto foi concebido com o objetivo de investigar como técnicas modernas de paralelização podem reduzir drasticamente o custo computacional de algoritmos clássicos de aprendizado de máquina.

---

## Organização do Repositório

Este repositório contém **todos os arquivos de código utilizados no desenvolvimento do projeto**, incluindo:

- Implementações em C  
- Kernels CUDA  
- Rotinas de paralelização com MPI e OpenMP  
- Arquivos auxiliares de experimentação e teste  

Além disso, o arquivo:

> **`knn_parallel.ipynb`**

reúne o **compilado de todo o projeto**, concentrando:
- a descrição conceitual da solução,  
- a estrutura do código,  
- os experimentos realizados,  
- as análises de desempenho e os principais resultados obtidos.

Esse notebook funciona como o documento central do projeto.

---

## Visão Geral da Solução

A implementação do KNN foi organizada em três estágios principais:

1. Cálculo paralelo das distâncias entre amostras de treino e teste  
2. Seleção dos *k* vizinhos mais próximos, utilizando estratégias eficientes de redução e seleção parcial  
3. Determinação da classe alvo, por meio de votação majoritária  

Cada uma dessas etapas foi cuidadosamente projetada para explorar ao máximo o paralelismo disponível no hardware, tanto na CPU quanto na GPU.

---

## Tecnologias e Conceitos Envolvidos
- Linguagem C de baixo nível  

- Arquitetura de GPUs NVIDIA (CUDA)  
- Paralelismo em memória compartilhada (OpenMP)  
- Computação distribuída (MPI)  
- Otimização de acesso à memória  
- Balanceamento de carga  
- Análise de desempenho e *speedup*

---

## Resultados e Impacto

Os experimentos demonstram ganhos significativos de desempenho quando comparados à versão sequencial do algoritmo, evidenciando o impacto direto da paralelização híbrida na eficiência do KNN.

O projeto fornece uma base sólida para estudos envolvendo:
- aprendizado de máquina em larga escala,  
- sistemas paralelos,  
- otimização de algoritmos computacionalmente intensivos.

---

## Autores

**Geovanne Mansano**, **Fernanda Nami**, **Giovanna Rosseto** e **Adriano Tavares**.  
Trabalho desenvolvido como projeto final de graduação em Ciência da Computação, com foco em **computação de alto desempenho**, **paralelização de algoritmos** e **ciência de dados em larga escala (Big Data)**.
