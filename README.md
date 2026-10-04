# 📊 Projeto Avaliativo: Pipeline Preditivo de Machine Learning

**Curso:** Machine Learning e Visão Computacional [T3]  
**Módulo:** 1 - Semana 14  
**Autor:** Marcio Baldissera Cure
**Vídeo de Apresentação:** [Link para o vídeo no Drive](https://drive.google.com/file/d/1HNFOvfC1hAkcW8oE711Bmyv_-GHbSASi/view?usp=sharing)

---

## 1. Contextualização e Problema de Negócio

Este projeto desenvolve um pipeline preditivo completo focado em resolver um problema real de negócios no setor Financeiro (Risco de Crédito).

**O Desafio:**  
Um banco precisa prever se um cliente se tornará inadimplente (alvo: `loan_status = 1`) ou se pagará o empréstimo em dia (alvo: `loan_status = 0`).

**Impacto Financeiro (O Custo do Erro):**  
Em nosso cenário, a Inteligência Artificial precisa ser avaliada sob a ótica financeira:
* **Falso Positivo:** A máquina classifica um bom pagador como "Risco de Calote". O banco deixa de conceder o crédito e perde a oportunidade de lucro com os juros, além de poder perder o cliente para a concorrência.
* **Falso Negativo:** A máquina classifica um mau pagador como "Seguro". O banco concede o crédito e sofre um calote, resultando em perda direta e imediata do capital emprestado. *Este é, de longe, o erro mais oneroso.*

---

## 2. Dicionário de Dados

A base de dados original conta com 32.581 unidades amostrais e 12 variáveis originais. Durante a fase de *Feature Engineering*, foi criada a variável estratégica exigida para enriquecer a análise.

| Variável | Descrição | Tipo de Dado |
| :--- | :--- | :--- |
| `person_age` | Idade do solicitante | `int` |
| `person_income` | Renda anual do solicitante | `int` |
| `person_home_ownership` | Situação de moradia (Alugada, Própria, etc.) | `object` |
| `person_emp_length` | Tempo de emprego em anos | `float` |
| `loan_intent` | Motivo/Intenção do empréstimo | `object` |
| `loan_grade` | Classificação do empréstimo | `object` |
| `loan_amnt` | Valor do empréstimo solicitado | `int` |
| `loan_int_rate` | Taxa de juros do empréstimo | `float` |
| `cb_person_default_on_file` | Histórico de inadimplência prévio (S/N) | `object` |
| `cb_person_cred_hist_length` | Tempo de histórico de crédito | `int` |
| **`comprometimento_renda`** | **(Nova Feature) `(loan_amnt / person_income) * 100`. Indica o % da renda comprometida.** | **`float`** |
| `loan_status` | **Variável Alvo (0 = Pagou, 1 = Calote)** | `int` |

---

## 3. Resumo Executivo: Insights da Análise Exploratória (EDA)

Durante a fase de exploração, extraímos diagnósticos cruciais que guiaram o tratamento e modelagem:

1. **Desbalanceamento Crítico:** A base possui dados com forte assimetria na variável alvo. Cerca de 78,18% (25.473) são bons pagadores (Classe 0) e apenas cerca de 21,82% (7.108) são inadimplentes (Classe 1).
2. **Correlações Relevantes:** O mapa de calor indicou forte relação colinear (0,86) entre a idade do indivíduo (`person_age`) e seu histórico de crédito (`cb_person_cred_hist_length`). Para o nosso alvo, a maior correlação relativa numérica encontrada foi com a variável calculada/percentual de renda (`loan_percent_income` com 0,38).
3. **Padrões por Intenção de Empréstimo:** Observou-se que a intenção do empréstimo influencia na inadimplência, sendo que consolidação de débito, gastos médicos e melhorias residenciais possuem taxas relativas maiores de inadimplência (cerca de 26-28%).

---

## 4. Metodologia: O Pipeline de Dados

O desenvolvimento seguiu o rigor estatístico necessário para evitar vazamento de dados (*Data Leakage*) e ruídos no aprendizado:

* **Limpeza Inicial:** Identificamos e removemos 165 linhas duplicadas da base.
* **Análise de Nulos e Outliers (Data Prep):** 
  * A variável `person_emp_length` possuía 887 dados nulos (2,73%) e discrepâncias graves (outliers impossíveis, como 123 anos de emprego). Transformamos valores acima de 35 anos em `NaN` e imputamos os valores nulos utilizando a **Mediana**, por ser menos sensível aos outliers restantes.
  * A variável `loan_int_rate` apresentou 3.095 nulos (9,54%) e também presença de outliers. Agrupei `loan_int_rate` por `loan_grade` e preenchi com a mediana de cada grupo.
* **Feature Engineering:** Criação da feature de `comprometimento_renda`, garantindo tratamento de nulos prévio.
* **Reanálise da matriz de correlação com nova variável:** Notamos que a nova variável é redundante com `loan_percent_income`, sendo assim, mantivemos a nova variável e removemos a antiga. 
* **Encoding e Split:** Conversão de variáveis categóricas usando One-Hot/Label Encoding e separação de treino/teste com 20% e `stratify=y` para manter a proporção das classes desbalanceadas.
* **Balanceamento:** Aplicação de SMOTE e UnderSampling estritamente nos dados de treino para evitar vazamento.
* **Escalonamento:** Uso de `StandardScaler` apenas para o modelo KNN. A Árvore de Decisão foi preservada sem escalonamento devido aos seus cortes monotônicos.

---

## 5. Modelagem e Diagnóstico de Overfitting

Foram testados dois algoritmos clássicos, variando seus hiperparâmetros de complexidade:

* **KNN (K-Nearest Neighbors):** Testados os valores $K \in \{3, 5, 7, 9\}$. 
* **Árvore de Decisão:** Testadas as profundidades $max\_depth \in \{3, 5, 7, None\}$.

**Diagnóstico:**  
*A Árvore de profundidade livre (None) apresentou Overfitting claro, com 100% no treino e queda vertiginosa no teste. A melhor generalização ocorreu para o KNN com K=9 balanceado por undersampling e Árvore com max_depth=7.*

---

## 6. Veredito de Negócios e Recomendação Final

Sob a ótica de negócios para concessão de crédito, nossa prioridade é evitar **Falsos Negativos** (dar crédito a quem vai dar calote), o que sugere a importância da métrica de **Recall** para a classe 1.

Olhando para a Matriz de Confusão:
* O modelo **[Vencedor: KNN]** obteve o melhor *Recall* (acerto de maus pagadores) após o balanceamento. 

**Recomendação para a Diretoria:**  
Recomenda-se colocar em produção o modelo **[KNN com k=9 e undersampling]**. Ele demonstrou melhor equilíbrio na Matriz de Confusão, minimizando significativamente a evasão de capital provocada por calotes imprevistos.

---

## 7. Como Executar o Projeto

**Pré-requisitos:** Python 3.8+ e Jupyter Notebook.

#### Clone este repositório:

   ```bash
   git clone https://github.com/Mauritia-flexuosa/projeto-avaliativo.git
   ```
