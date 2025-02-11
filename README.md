# Projeto de Carro Autônomo com Deep Q-Learning

Este projeto é do curso [Aprendizagem por Reforço com Deep Learning, PyTorch e Python](https://www.udemy.com/course/aprendizagem-reforco-deep-learning-pytorch-python/), do professor **Jones Granatyr** e a **IA Expert Academy**. O objetivo é treinar um carro autônomo para navegar em um mapa, evitando obstáculos e alcançando um objetivo específico. O carro é controlado por uma rede neural que utiliza a técnica de Deep Q-Learning para aprender a melhor política de navegação.

## Estrutura do Projeto

### Descrição dos Arquivos

- [`ai.py`](ai.py): Implementa a rede neural e o algoritmo de Deep Q-Learning.
- [`car.kv`](car.kv): Define a interface gráfica do carro e dos sensores usando Kivy.
- [`last_brain.pth`](last_brain.pth): Arquivo de checkpoint para salvar e carregar o estado da rede neural.
- [`map.py`](map.py): Implementa a lógica do mapa e do carro autônomo, incluindo a interface gráfica e a interação com o usuário.

## Instalação

1. Clone o repositório:
    ```bash
    git clone <URL do repositório>
    cd <nome do repositório>
    ```

2. Crie um ambiente virtual e instale as dependências:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

    ou se estiver utilizando anaconda

    ``` bash
    conda create -n nome_do_ambiente python=3.6.13
    conda activate nome_do_ambiente
    pip install -r requirements.txt
    ```

## Uso

1. Execute o script principal para iniciar a simulação:
    ```bash
    python map.py
    ```

2. A interface gráfica será aberta, mostrando o carro autônomo e o mapa. O carro começará a se mover e aprender com suas ações.

## Estrutura do Código

### Rede Neural

A rede neural é definida na classe `Network` em [ai.py](ai.py). Ela possui três camadas totalmente conectadas (fully connected):

O Input_size (5 ações) -> 50 neurônios (camada oculta) -> 30 neurônios (saída da camada oculta) -> 3 (saída da rede)

### Experiência de Replay

A fim de evitar overfitting e acelerar o treinamento, é implementada a *Replay Experience* com capacidade máxima de 100.000 transições, selecionando 100 delas para o treinamento.
        
### Atualização dos Pesos

Para a atualização dos pesos, são obtidos os valores de Q para o estado atual e o próximo. Com os valores de Q do próximo estado, é calculado o valor do *alvo*, e com ele a função de custo. Por fim, são atualizados os gradientes com base na perda.

### Escolha da Próxima Ação

Para a escolha da próxima ação, é submetido o estado atual à rede neural para obter os valores de Q para todas as possibilidades. Aplicando o conceito de *Exploration X Exploitation*, usando a função softmax, é garantido que o agente explore o ambiente, podendo achar uma solução melhor do que a que já possui.