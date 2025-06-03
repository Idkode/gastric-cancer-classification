# Gastric Cancer Classification

Este repositório se refere a um trabalho em andamento para um problema de classificação multiclasse em histopatologia do câncer gástrico

## Dataset
**Nome**: HMU-GC-HE-30K – *Gastric Cancer Histopathology Tissue Image Dataset (GCHTID)*  
**Fonte**: Lou *et al.* “A large histological images dataset of gastric cancer with tumour microenvironment annotation for AI”. DOI: 10.1038/s41597-025-04489-9.
Disponível em https://www.nature.com/articles/s41597-025-04489-9.

**Conteúdo**: ~31.000 recortes de imagens RGB (224 × 224 px) extraídos de 300 lâminas histológicas digitalizadas (WSI) de pacientes com câncer gástrico, coradas com H&E.

Cada recorte é anotado com uma das oito classes de tecido do microambiente tumoral (TME):

| Abrev. | Tecido / Componente    |
|-------|-------------------------|
| **ADI** | Adipose tissue        |
| **DEB** | Debris                |
| **MUC** | Mucus                 |
| **MUS** | Muscle                |
| **LYM** | Lymphocyte aggregates |
| **STR** | Stroma                |
| **NOR** | Normal mucosa         |
| **TUM** | Tumour epithelium     |


## Detalhes dos Experimentos

A tarefa realizada será a classificação das imagens do dataset nas 8 classes de Tecido/Componente listadas acima. A arquitetura base escolhida para o treinamento foi a ResNet152 pré-treinada com os pesos IMAGENET1K_V1 disponibilizados pela biblioteca Torch.  Como essa rede teve um bom desempenho na competição ImageNet, com muitas classes diferentes, acreditamos que sua capacidade de extração de características seja útil para a resolução do problema.
A partir dessa arquitetura, serão realizados experimentos em 5 arquiteturas diferentes da rede _fully connected_. Para os quatro experimentos, serão mantidos os seguintes parâmetros: otimizador Adam, função de perda CrossEntropyLoss, número de épocas de 800, _early stopping_ com paciência de 80 épocas,  scheduler de redução do learning rate em 10% com uma paciência de 6 épocas e batch de 64.
Os blocos finais da rede layer3, layer4 e avgpool serão descongelados para que sejam ajustados ao dataset e características específicas dele possam ser aprendidas.

| Parâmetro                | Valor                                  |
|------------------------------|----------------------------------------|
| **Otimizador**               | Adam                                   |
| **Função de perda**          | CrossEntropyLoss                       |
| **Número de Épocas**         | 800                                    |
| **Estratégia de Validação**  | Holdout                                |
| **Partição de Teste**        | 20%                                    |
| **Partição de Validação**    | 10% do restante (após teste)           |
| **Partição de Treino**       | 90% do restante (após teste)           |
| **Paciência**                | 80                                     |
| **Scheduler**                | Redução do learning rate em 10%        |
| **Paciência do scheduler**   | 6 épocas                               |    
| **Batch size**               | 64                                     |
| **Blocos descongelados**     | layer3, layer4, avgpool                |

## Particionamento do Dataset
O dataset foi particionado conforme implementação dos autores, sendo 20% do dataset utilizado para teste, 10% do restante para validação e 90% do restante para treino.

| Conjunto       | Porcentagem do total |
|----------------|----------------------|
| **Teste**      | 20%                  |
| **Treino**     | 72%                  |
| **Validação**  | 8%                   |


## Arquiteturas propostas
As cinco arquiteturas são:

- Número de neurônios decrescente:
    - Linear(model.fc.in_features, 2048)
    - ReLU
    - Linear(2048, 512)
    - ReLU
    - Linear(512, 64)
    - ReLU
    - Linear(64, 8)

- FC_1024_256_BN_Dropout:
    - Linear(model.fc.in_features, 1024)
    - BatchNorm1d(1024)
    - ReLU
    - Dropout(0.4)
    - Linear(1024, 256)
    - BatchNorm1d(256)
    - ReLU
    - Dropout(0.4)
    - Linear(256, 8)

- FC_2048_512_GELU:
    - Linear(model.fc.in_features, 2048)
    - GELU
    - Linear(2048, 512)
    - GELU
    - Linear(512, 8)
