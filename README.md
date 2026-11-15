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

## Etapa 2 · Simulação e detecção de drift

A etapa simula seis meses de produção sob um choque econômico, pontua cada lote com o modelo da etapa 1 e compara cada mês com a referência de treino usando o Evidently AI. O objetivo é mostrar quais variáveis mudaram, quando a mudança começou e quanto o modelo perdeu de desempenho.

### Dataset de produção

A produção parte dos 6.000 registros de avaliação da etapa 1, que o modelo não viu no treino. Eles são embaralhados com semente 42 e divididos em seis lotes mensais de 1.000 registros. O choque cresce de forma linear: intensidade 0 no mês 1 e intensidade total no mês 6. O mês 1 sai inalterado e serve de controle para falsos alarmes.

| Fenômeno | Variáveis alteradas | Mudança no mês 6 |
| --- | --- | --- |
| Inflação e maior uso do crédito rotativo | `bill_amount_m1` a `bill_amount_m3` | Faturas nominais 80% maiores. Os pagamentos ficam iguais, pois a renda não acompanha. |
| Novo perfil de clientes | `age` e `credit_limit` | Cada cliente tem 50% de chance de vir do novo perfil, com idade de 19 a 29 anos e limite reduzido à metade, em múltiplos de 10.000. |
| Mudança de conceito | `default_next_month` | Cada adimplente com fatura de pelo menos 70% do limite tem 30% de chance de passar a inadimplir. |

As faturas são a segunda variável mais influente do modelo. Na regressão padronizada, `bill_amount_m1` tem coeficiente −0,32, atrás apenas de `payment_status_m1`, com 0,60. O sinal negativo faz o modelo enxergar menos risco quando a fatura sobe. O status e os valores de pagamento ficam inalterados e servem de controle entre as variáveis.

Todos os lotes passam pelo contrato da etapa 1 antes de serem gravados. O contrato garante tipos e faixas, mas não percebe mudança de distribuição. Essa detecção cabe ao monitoramento de drift. Os parâmetros do cenário ficam em `artifacts/production/cenario.json`.

### Como rodar

Depois do treino da etapa 1, que gera `artifacts/reference.csv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m credit_scoring simulate
.\.venv\Scripts\python.exe -m credit_scoring drift
```

O comando `drift` valida todos os lotes de `artifacts/production` antes de pontuar. Um lote que viola o contrato interrompe a execução com código `2`, sem gravar relatório ou predições.

| Saída | Conteúdo |
| --- | --- |
| `reports/drift_report.html` | Relatório Evidently do mês 6: tabela PSI, tabela Kolmogorov-Smirnov e qualidade da classificação contra a referência. |
| `reports/drift_timeline.csv` | PSI, p-valor do KS e severidade por variável e por mês. |
| `reports/drift_summary.json` | Regras usadas e, por mês, desempenho do modelo, variáveis com drift e alerta de desempenho. |
| `artifacts/production/mes_01.csv` a `mes_06.csv` | Lotes de produção, disponíveis localmente. |
| `artifacts/production/predicoes.csv` | Score e classe prevista para cada cliente de cada mês, disponível localmente. |

### Regras de detecção

Cada variável e o score do modelo passam por dois testes do Evidently. O PSI mede o tamanho da mudança, e o teste de Kolmogorov-Smirnov mede se ela é estatisticamente significativa. A variável só é marcada com drift quando os dois concordam: PSI de pelo menos 0,1 e p-valor do KS abaixo de 0,05. O KS sozinho acusa diferenças pequenas em amostras grandes. No mês 6, por exemplo, ele marca o score do modelo com p-valor de 0,0001, embora o PSI seja de apenas 0,05.

| Severidade | Critério |
| --- | --- |
| Estável | PSI abaixo de 0,1 ou p-valor do KS de pelo menos 0,05. |
| Moderado | Drift com PSI de 0,1 até menos de 0,25. |
| Forte | Drift com PSI de 0,25 ou mais. |

O relatório HTML marca drift no lote quando pelo menos 30% das colunas monitoradas mudam, abaixo dos 50% padrão do Evidently. Em crédito, uma mudança dessa proporção já pede revisão do modelo. O alerta de desempenho dispara quando a ROC AUC fica mais de 0,03 abaixo do baseline da etapa 1, que é 0,7063, ou quando o recall de inadimplência fica mais de 10 pontos percentuais abaixo de 62,32%.

### Resultados

| Mês | Variáveis com drift | Inadimplência real | Inadimplência prevista | ROC AUC | Recall | Alerta de desempenho |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | nenhuma | 21,2% | 37,1% | 0,705 | 61,8% | não |
| 2 | nenhuma | 24,0% | 34,6% | 0,735 | 61,3% | não |
| 3 | `age` | 25,6% | 33,9% | 0,711 | 59,0% | não |
| 4 | `age` e faturas `m1` a `m3` | 28,7% | 33,5% | 0,696 | 53,7% | não |
| 5 | `age` e faturas `m1` a `m3` | 32,5% | 30,5% | 0,655 | 44,6% | sim |
| 6 | `age`, `credit_limit` e faturas `m1` a `m3` | 35,7% | 31,3% | 0,672 | 47,6% | sim |

No mês 6, o relatório do Evidently aponta cinco variáveis degradadas:

| Variável | PSI | p-valor do KS | Severidade |
| --- | ---: | ---: | --- |
| `age` | 0,620 | 7,3 × 10⁻⁹² | forte |
| `bill_amount_m2` | 0,313 | 1,1 × 10⁻²⁸ | forte |
| `bill_amount_m1` | 0,284 | 3,1 × 10⁻²⁶ | forte |
| `bill_amount_m3` | 0,232 | 4,5 × 10⁻²⁸ | moderado |
| `credit_limit` | 0,145 | 7,0 × 10⁻²⁰ | moderado |

As seis variáveis de controle ficam com PSI de no máximo 0,03. O mês 1 não gera nenhum alerta.

A degradação é silenciosa. A inadimplência real sobe de 21,2% para 35,7%, mas a taxa prevista pelo modelo cai de 37,1% para 31,3%, e o score muda pouco (PSI de 0,05). Quem monitorasse só as predições não veria o problema. O drift das variáveis aparece no mês 3, dois meses antes de os rótulos confirmarem a queda de desempenho. No mês 6, o modelo deixa passar 187 dos 357 inadimplentes.

As métricas de referência exibidas no HTML são calculadas sobre os dados de treino e por isso são mais otimistas, com ROC AUC de 0,726. A comparação oficial usa o baseline medido na avaliação separada da etapa 1.
