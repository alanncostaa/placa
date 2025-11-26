# **📘 README COMPLETO — PLACA2 (Atualizado com dados OpenALPR)**

`````markdown
# 🚗 PLACA2 — Reconhecimento de Placas (ANPR/OCR)

Este projeto implementa um sistema completo de Reconhecimento Automático de Placas Veiculares (ANPR/OCR), utilizando:

- **OpenCV** para pré-processamento e segmentação de caracteres  
- **Scikit-learn** para classificação (KNN, SVM e Random Forest)  
- Dataset real do **OpenALPR Benchmark**, com placas norte-americanas

O objetivo é demonstrar uma pipeline funcional de OCR para placas, desde o pré-processamento até a leitura final.

---

# 🗂️ Estrutura do Projeto

````plaintext
PLACA2/
├── dataset/
│   ├── train/              # Imagens originais das placas (treino)
│   ├── test/               # Imagens de teste (opcional)
│   └── labels.csv          # Metadados + rótulos das placas
│
├── dataset_chars/          # Criada pelo script de segmentação
│                           # Armazena caracteres recortados por classe
│
├── outputs/
│   └── models/             # Modelos treinados (.pkl): KNN, SVM, RF
│
├── src/
│   ├── segment_chars.py    # Segmenta placa em caracteres e monta dataset_chars
│   ├── train_chars.py      # Treina os classificadores e salva modelos
│   ├── test_plate.py       # Testa leitura de uma placa nova
│   ├── preprocess.py       # Funções de pré-processamento (blur, binarização etc.)
│   └── utils.py            # Funções auxiliares
│
└── requirements.txt        # Dependências Python
`````

---

# 📥 **PASSO PRÉVIO — Baixando e preparando o dataset**

### 1️⃣ Baixe o dataset oficial do OpenALPR:

📎 **Link:** [https://github.com/openalpr/benchmarks](https://github.com/openalpr/benchmarks)

### 2️⃣ Baixe o arquivo `.zip` do repositório

Você encontra as imagens dentro da pasta:

```
seg_and_ocr/usimages/
```

### 3️⃣ Copie todas as imagens dessa pasta para:

```
PLACA2/dataset/train/
```

### 4️⃣ Certifique-se de que existe um arquivo:

```
PLACA2/dataset/labels.csv
```

Com as colunas mínimas:

| filename           | plate   |
| ------------------ | ------- |
| nome_da_imagem.jpg | ABC1234 |

---

# 🚀 Instalação

### Criar ambiente virtual (opcional)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### Instalar dependências

```bash
pip install -r requirements.txt
```

---

# 📋 Execução — Passo a Passo

## 1️⃣ **Segmentar os caracteres das placas**

Este script:

* Lê imagens de `dataset/train`
* Segmenta cada caractere
* Cria a pasta `dataset_chars/`
* Organiza por classe (A, B, C, 0, 1, 2…)

```bash
python src/segment_chars.py
```

📌 *Saída:*
`dataset_chars/` contendo todas as pastas de caracteres.

---

## 2️⃣ **Treinar os modelos (KNN, SVM, RF)**

```bash
python src/train_chars.py
```

📌 *Saída:*
Modelos gerados dentro de `outputs/models/`:

* `knn_chars.pkl`
* `svm_chars.pkl`
* `rf_chars.pkl`

Além das métricas impressas no terminal.

---

## 3️⃣ **Testar uma placa nova**

```bash
python src/test_plate.py --image caminho/para/placa.jpg
```

O script realiza:

* Segmentação da placa
* Classificação caractere por caractere
* Montagem final da placa reconhecida

📌 *Saída:*
O terminal exibe algo como:

```
KNN: ABC1234
SVM: ABC1234
RF:  ABC1234
```

---

# 📌 Observações

* As pastas `outputs/` e `dataset_chars/` são geradas automaticamente.
* Placas do dataset OpenALPR são dos EUA — o formato de caracteres pode variar.
* Não envie arquivos `.pkl` para o GitHub (acima de 100MB podem causar erro).

---



É só pedir!
```

