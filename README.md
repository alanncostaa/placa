## 📄 README.md (Reconhecimento de Placas de Carro - OCR)

```markdown
# 🚗 Reconhecimento de Placas de Carro (ANPR/OCR)

Este projeto implementa um sistema básico de Reconhecimento Óptico de Caracteres (OCR) para placas de veículos, utilizando técnicas de Visão Computacional (OpenCV) para segmentação e Machine Learning (Scikit-learn) para classificação de caracteres.

## 🗂️ Estrutura do Projeto

Abaixo está a estrutura principal do projeto:

```

PLACA2/
├── dataset/
│   ├── train/                 \# Contém as imagens originais da placa.
│   ├── test/                  \# Contém as imagens de teste da placa (opcional).
│   └── labels.csv             \# Arquivo CSV com metadados e o texto da placa (rótulo).
├── dataset\_chars/             \# Gerada pelo script de segmentação. Armazena os caracteres recortados e agrupados por classe para o treinamento.
├── outputs/
│   └── models/                \# Modelos ML treinados (.pkl) são salvos aqui (KNN, SVM, RF).
├── src/
│   ├── segment\_chars.py       \# Segmenta caracteres das placas e organiza o 'dataset\_chars'.
│   ├── train\_chars.py         \# Treina os modelos de classificação (KNN, SVM, RF) e avalia métricas.
│   ├── test\_plate.py          \# Carrega os modelos, faz segmentação e lê a placa na imagem de teste.
│   ├── preprocess.py          \# Funções de pré-processamento de imagem.
│   └── utils.py               \# Funções auxiliares (leitura de imagem, etc.).
└── requirements.txt           \# Dependências Python necessárias.

````

## 🚀 Instalação e Configuração

### 1. Requisitos

Certifique-se de ter o Python (3.x) instalado.

```bash
# Crie e ative o ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows
````

### 2\. Instalação de Dependências

Instale todas as bibliotecas necessárias usando o `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 📋 Passo a Passo para Execução

Siga os passos abaixo para treinar o modelo e testar a leitura da placa.

### ⚠️ **PASSO PRÉVIO: Preparação dos Dados**

Antes de iniciar, certifique-se de que seus dados estão no local correto:

1.  **Imagens:** Coloque as imagens das placas que serão usadas para treinamento dentro da pasta `dataset/train/`.
2.  **Rótulos (`labels.csv`):** O arquivo `dataset/labels.csv` deve estar preenchido, contendo, no mínimo, as colunas:
      * `filename`: Nome do arquivo da imagem da placa.
      * `plate`: O texto da placa (o rótulo correto).

### 1\. Segmentação de Caracteres

Este passo usa as imagens em `dataset/train` para recortar cada caractere individualmente e agrupá-los em classes (pastas nomeadas 'A', 'B', '0', '1', etc.) dentro de `dataset_chars`.

```bash
python src/segment_chars.py
```

> **Resultado:** A pasta `dataset_chars/` será criada/atualizada com os subdiretórios de cada classe.

### 2\. Treinamento e Avaliação dos Modelos

Este passo carrega os caracteres de $28 \times 28$ pixels da pasta `dataset_chars`, treina os modelos de classificação (KNN, SVM, Random Forest) e avalia suas métricas (Acurácia, F1-Score, Precisão, etc.).

```bash
python src/train_chars.py
```

> **Resultado:** Os modelos treinados (`knn_chars.pkl`, `svm_chars.pkl`, `rf_chars.pkl`) serão salvos em `outputs/models/`.

### 3\. Teste de Leitura da Placa

Após o treinamento, você pode testar a capacidade do sistema de ler uma placa em uma nova imagem, executando a segmentação e a classificação em tempo real.

```bash
# Substitua 'caminho/para/sua/imagem.jpg' pelo caminho real da imagem de teste.
python src/test_plate.py --image "caminho/para/sua/imagem.jpg"
```

> **Resultado:** O console exibirá as previsões da placa para cada um dos modelos treinados (KNN, SVM, RF).

```
```