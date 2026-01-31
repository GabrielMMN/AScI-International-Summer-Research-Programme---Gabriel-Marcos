import sys
import numpy as np
import scipy.spatial
import time

class CData:
    def __init__(self, matricula, ni, nj, p):
        np.random.seed(matricula)
        self.p, self.ni, self.nj = p, ni, nj
        self.d = np.random.randint(100, size=(ni,))
        self.xyi = np.random.randint(300, size=(ni, 2))
        self.xyj = np.random.randint(300, size=(nj, 2))
        self.c = np.ceil(scipy.spatial.distance.cdist(self.xyi, self.xyj))

def calcular_funcao_objetivo(d, c, s, ni, nj):
    return sum(d[i] * min(c[i, j] for j in range(nj) if s[j] == 1) for i in range(ni))

def alocar_clientes(d, c, s, ni, nj):
    clientes_por_facilidade = {j: [] for j in range(nj) if s[j] == 1}
    demanda_por_facilidade = {j: 0 for j in clientes_por_facilidade}
    
    for i in range(ni):
        j_selecionado = min(clientes_por_facilidade.keys(), key=lambda j: c[i, j])
        clientes_por_facilidade[j_selecionado].append(i + 1)  # Clientes numerados de 1 a ni
        demanda_por_facilidade[j_selecionado] += d[i]
    
    return clientes_por_facilidade, demanda_por_facilidade

def imprimir_resultado(nome_heuristica, custo, tempo_execucao, s, d, c, ni, nj):
    clientes_por_facilidade, demanda_por_facilidade = alocar_clientes(d, c, s, ni, nj)
    
    print(f"{nome_heuristica}:")
    print(f"Custo total              : {custo:10.2f}.")
    print(f"Tempo execucao           : {tempo_execucao:10.2f} s.")
    print("facilidades : demanda : clientes")
    
    for j, clientes in sorted(clientes_por_facilidade.items()):
        print(f"{j:10} : {demanda_por_facilidade[j]:7} : {' '.join(map(str, clientes))}")
    print()

def solucao_aleatoria(nj, p):
    s = np.zeros(nj, dtype=int)
    indices = np.random.choice(nj, p, replace=False)
    s[indices] = 1
    return s

def solucao_gulosa(d, c, ni, nj, p):
    s = np.zeros(nj, dtype=int)
    demanda_atendida = np.zeros(ni, dtype=bool)
    
    while sum(s) < p:
        i_maior = np.argmax(d * ~demanda_atendida)
        j_melhor = np.argmin(c[i_maior, :])
        s[j_melhor] = 1
        demanda_atendida[i_maior] = True
    return s

def refinamento_troca_1bit(d, c, s, ni, nj):
    custo_atual = calcular_funcao_objetivo(d, c, s, ni, nj)
    melhorou = True
    
    while melhorou:
        melhorou = False
        facilidades_ativas = np.where(s == 1)[0]
        facilidades_inativas = np.where(s == 0)[0]
        
        for j_out in facilidades_ativas:
            for j_in in facilidades_inativas:
                s_temp = s.copy()
                s_temp[j_out], s_temp[j_in] = 0, 1
                if sum(s_temp) != sum(s):
                    continue
                
                novo_custo = calcular_funcao_objetivo(d, c, s_temp, ni, nj)
                if novo_custo < custo_atual:
                    s, custo_atual = s_temp, novo_custo
                    melhorou = True
                    break
            if melhorou:
                break
    return s

#def refinamento_troca_2bits(d, c, s, ni, nj):
    #return refinamento_troca_2bits(d, c, s, ni, nj)

def main():
    assert len(sys.argv) > 4, 'please, provide <matricula> <ni> <nj> <p>'
    matricula, ni, nj, p = [int(val) for val in sys.argv[1:]]
    dt = CData(matricula, ni, nj, p)
    
    for nome, funcao in [
        ("Solução Aleatória", solucao_aleatoria),
        ("Solução Gulosa", solucao_gulosa),
    ]:
        start = time.time()
        s = funcao(dt.d, dt.c, ni, nj, p) if funcao == solucao_gulosa else funcao(nj, p)
        custo = calcular_funcao_objetivo(dt.d, dt.c, s, ni, nj)
        tempo_execucao = time.time() - start
        imprimir_resultado(nome, custo, tempo_execucao, s, dt.d, dt.c, ni, nj)
        
        for ref_nome, ref_func in [
            ("Refinamento 1", refinamento_troca_1bit),
            #("Refinamento 2", refinamento_troca_2bits),
        ]:
            start = time.time()
            s_ref = ref_func(dt.d, dt.c, s, ni, nj)
            custo_ref = calcular_funcao_objetivo(dt.d, dt.c, s_ref, ni, nj)
            tempo_ref = time.time() - start
            imprimir_resultado(f"{ref_nome} ({nome})", custo_ref, tempo_ref, s_ref, dt.d, dt.c, ni, nj)

if __name__ == "__main__":
    main()