import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perfprof import perfprof
import inspect

# Verificar a assinatura da função perfprof
print("Assinatura da função perfprof:")
print(inspect.signature(perfprof))

# Carregar os dados
df = pd.read_csv('dados.csv', sep=';')

# Verificar os dados
print("\nEstrutura dos dados carregados:")
print(f"Dimensões do DataFrame: {df.shape}")
print("Colunas:", df.columns.tolist())
print("Dados:")
print(df.head(11))

# Remover a coluna 'instância' se existir
if 'instância' in df.columns:
    df = df.drop('instância', axis=1)

# Obter nomes dos algoritmos
algorithms = df.columns.tolist()
num_algorithms = len(algorithms)
print(f"\nAlgoritmos detectados ({num_algorithms}): {algorithms}")

# Converter para matriz numpy
data = df.values.T  # Transpor para que cada linha represente um algoritmo
print(f"Formato da matriz de dados: {data.shape}")

# Criar uma lista de estilos de linha GARANTINDO que temos exatamente 1 para cada algoritmo
line_styles = ['-', '--', ':', '-.', (0, (3, 1)), (0, (1, 1))]
print(f"Número de estilos de linha originais: {len(line_styles)}")

# Se isso não for suficiente, vamos adicionar mais estilos
while len(line_styles) < num_algorithms:
    line_styles.append('-')

print(f"Número de estilos de linha após ajuste: {len(line_styles)}")
print(f"Estilos de linha: {line_styles}")

# Alternativa: tentar implementar manualmente o performance profile
print("\nTentando método alternativo...")

try:
    # Normalizar os dados para maximização
    # Para cada problema, dividir cada valor pelo melhor valor
    ratios = np.zeros_like(data, dtype=float)
    
    for j in range(data.shape[1]):  # Para cada problema/instância
        best_val = np.max(data[:, j])  # Melhor valor para este problema
        for i in range(data.shape[0]):  # Para cada algoritmo
            ratios[i, j] = data[i, j] / best_val if best_val != 0 else 1.0
    
    # Calcular a distribuição cumulativa
    tau_values = np.linspace(0.5, 1.0, 100)  # Valores de tau de 0.5 a 1.0
    profiles = np.zeros((len(algorithms), len(tau_values)))
    
    for i in range(len(algorithms)):  # Para cada algoritmo
        for k, tau in enumerate(tau_values):  # Para cada valor de tau
            profiles[i, k] = np.mean(ratios[i, :] >= tau)
    
    # Plotar o performance profile
    plt.figure(figsize=(10, 6))
    
    for i, algo in enumerate(algorithms):
        plt.plot(tau_values, profiles[i], label=algo, linestyle=line_styles[i])
    
    plt.xlabel('Tau (τ) - Fator de performance')
    plt.ylabel('Fração de problemas')
    plt.title('Performance Profile dos Algoritmos')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Salvar e mostrar o gráfico
    plt.savefig('performance_profile_manual.png', dpi=300)
    plt.show()
    
    print("Gráfico criado com sucesso usando o método alternativo!")
    
except Exception as e:
    print(f"Erro no método alternativo: {e}")

# Análise adicional
print("\nEstatísticas por algoritmo:")
for i, algo in enumerate(algorithms):
    values = data[i]
    print(f"{algo}: Média = {np.mean(values):.2f}, Melhor = {np.max(values)}, Pior = {np.min(values)}")

# Contar quantas vezes cada algoritmo foi o melhor
best_counts = {algo: 0 for algo in algorithms}
for j in range(data.shape[1]):  # Para cada problema
    best_value = np.max(data[:, j])  # Melhor valor para este problema
    
    for i, algo in enumerate(algorithms):
        if data[i, j] == best_value:
            best_counts[algo] += 1

print("\nNúmero de vezes que cada algoritmo foi o melhor:")
for algo, count in best_counts.items():
    print(f"{algo}: {count} vezes")