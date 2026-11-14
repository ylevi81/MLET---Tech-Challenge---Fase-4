# Tech Challenge · Fase 4

## Etapa 1 · Validação de dados e contratos

O projeto valida lotes de dados de crédito com Pandera e treina uma regressão logística para estimar inadimplência no mês seguinte. O alvo `default_next_month` vale `1` para inadimplência e `0` para os demais casos.

### Base e variáveis

Foi usada a base **Default of Credit Card Clients**, de I-Cheng Yeh, com 30.000 registros de clientes de cartão de crédito em Taiwan. Os valores monetários são em dólares taiwaneses. A base tem licença CC BY 4.0. [UCI Machine Learning Repository · DOI 10.24432/C55S3H](https://doi.org/10.24432/C55S3H)

O modelo usa idade, limite de crédito, status de pagamento, valor faturado e valor pago nos três meses anteriores. O identificador serve para verificar duplicatas e ordenar os dados antes da divisão. Sexo, escolaridade e estado civil foram deixados fora do modelo. A base de origem é histórica e os resultados se aplicam à avaliação acadêmica nesse conjunto.

### Como rodar

Na raiz do repositório, com Python 3.13 e PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m credit_scoring train
```

O treino baixa o XLS da UCI para `data/credit_default.xls` se a cópia local ainda não existir. Para usar outro arquivo local, passe `train --input caminho/do/arquivo.xls`. O treino também aceita um CSV com os nomes de coluna do contrato.

O script valida o lote inteiro antes do treino. A divisão estratificada usa 80% dos registros como referência de treinamento e 20% para avaliação separada, com semente 42. A normalização aprende os parâmetros apenas na referência. A regressão logística usa pesos de classe balanceados e as métricas de classificação usam limiar `0.5`.

| Saída | Conteúdo |
| --- | --- |
| `artifacts/baseline.joblib` | Pipeline treinado de normalização e regressão logística. |
| `artifacts/metrics.json` | Métricas da avaliação, incluindo ROC AUC, precisão média, recall e matriz de confusão. |
| `artifacts/reference.csv` | Linhas validadas usadas no treino, disponíveis localmente para a etapa de drift. |

Na execução com a base da UCI, 24.000 registros serviram de referência e 6.000 de avaliação. No conjunto de avaliação, o modelo obteve ROC AUC de **0,7063**, precisão média de **0,4848**, acurácia balanceada de **0,6607** e recall de inadimplência de **62,32%**.

### Contrato de dados

| Campo | Regra |
| --- | --- |
| `customer_id` | Inteiro positivo e único dentro do lote. |
| `age` | Inteiro entre 19 e 120 anos. |
| `credit_limit` | Inteiro positivo. A fonte tem limite de crédito como variável financeira. |
| `payment_status_m1` a `payment_status_m3` | Inteiros entre −2 e 9. `m1` é o mês mais recente. |
| `bill_amount_m1` a `bill_amount_m3` | Inteiros. Valores negativos podem representar saldo credor na fatura. |
| `payment_amount_m1` a `payment_amount_m3` | Inteiros não negativos. |
| `default_next_month` | Inteiro binário. Obrigatório no treino e ausente em lotes para pontuação. |

O contrato rejeita lotes vazios, campos nulos, tipos incompatíveis e colunas ausentes, extras ou duplicadas. No XLS original, a leitura projeta as variáveis escolhidas e usa os nomes do contrato. Para validar um CSV para pontuação, use `validate caminho/do/lote.csv --sem-alvo`.

### Demonstração de bloqueio

O lote de exemplo contém uma idade abaixo do mínimo, um limite de crédito ausente e um identificador repetido. Rode os comandos abaixo para ver a rejeição na validação e na entrada do treino.

```powershell
.\.venv\Scripts\python.exe -m credit_scoring validate examples/lote_invalido.csv
$LASTEXITCODE
.\.venv\Scripts\python.exe -m credit_scoring train --input examples/lote_invalido.csv --output-dir artifacts/lote_invalido
$LASTEXITCODE
Test-Path artifacts/lote_invalido
```

O código `2` indica rejeição pelo contrato. Em uma execução a partir de uma pasta de saída inexistente, `Test-Path` retorna `False`, pois o treino é interrompido antes de gravar qualquer artefato.
