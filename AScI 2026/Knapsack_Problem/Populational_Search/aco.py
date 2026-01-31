import os
import sys
import numpy as np
import time
from mip import Model, xsum, maximize, BINARY, OptimizationStatus, CBC

#np.random.seed(0)

class CInstance:
    def __init__(self, filename):
        self.read_file(filename)

    def read_file(self, filename):
        assert os.path.isfile(filename), 'Arquivo de dados inválido.'
        with open(filename, 'r') as rf:
            lines = [line.strip() for line in rf.readlines() if line.strip()]
            self.n = int(lines[0])
            self.b = int(lines[1])
            self.p = np.zeros(self.n, dtype=int)
            self.w = np.zeros(self.n, dtype=int)
            for i in range(self.n):
                self.p[i], self.w[i] = map(int, lines[2 + i].split())
        self.M = self.p.sum()

class CSolution:
    def __init__(self, inst):
        self.inst = inst
        self.x = np.zeros(inst.n, dtype=int)    # vetor binário que representa uma solução candidata
        self.obj = 0
        self._b = 0

    def evaluate(self):
        self._b = np.dot(self.x, self.inst.w)   # Calcula o peso total da solução
        self.obj = np.dot(self.x, self.inst.p)  # Calcula o valor da solução
        if self._b > self.inst.b:               # REDUNDANTE se o peso total da soluçao excede a capacidade da mochila, então FO = 0
            self.obj = 0
        return self.obj

    def copy(self, other):                      # armazena uma cópia da solução
        self.x[:] = other.x[:]
        self.obj = other.obj
        self._b = other._b

class CACO:
    def __init__(self, inst, alpha=0.865, beta=4.592, rho=0.098, Q=20, max_iter_sem_melhora=106):    # Parâmetros
        self.inst = inst
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.max_iter_sem_melhora = max_iter_sem_melhora
        self.n_formigas = inst.n
        self.tau = np.full(inst.n, 0.547)

    def construir_solucao(self):
        inst = self.inst
        formiga = CSolution(inst)
        itens_disponiveis = set(range(inst.n))      # Conjunto de itens ainda não selecionados

        while True:
            candidatos = [j for j in itens_disponiveis if formiga._b + inst.w[j] <= inst.b]     # Restrição de capacidade
            if not candidatos:
                break

            heuristicas = np.array([inst.p[j] for j in candidatos], dtype=float)
            feromonios = np.array([self.tau[j] for j in candidatos], dtype=float)
            atratividade = (feromonios ** self.alpha) * (heuristicas ** self.beta)
            soma = np.sum(atratividade)
            probs = atratividade / soma                     # Probabilidade de seleção de cada item

            j = np.random.choice(candidatos, p=probs)       # Roleta
            formiga.x[j] = 1
            formiga._b += inst.w[j]
            itens_disponiveis.remove(j)                     # Atualiza os itens disponíveis

        formiga.evaluate()
        return formiga

    def atualizar_feromonio(self, melhor_sol):
        self.tau *= (1 - self.rho)                          # Evaporação do feromônio
        for j in range(self.inst.n):
            if melhor_sol.x[j] == 1:
                self.tau[j] += self.Q * self.inst.p[j]      # Reforço do feromônio p/ itens da melhor solução

    def run(self):                                          # Execução do algoritmo
        print('Executando o algoritmo ACO...')
        melhor_sol = CSolution(self.inst)
        iter_sem_melhora = 0
        geracao = 0
        tempo_inicio = time.time()

        while iter_sem_melhora < self.max_iter_sem_melhora:
            geracao += 1
            melhores_geracao = [self.construir_solucao() for _ in range(self.n_formigas)]
            melhor_geracao = max(melhores_geracao, key=lambda sol: sol.obj)

            if melhor_geracao.obj > melhor_sol.obj:
                melhor_sol.copy(melhor_geracao)
                iter_sem_melhora = 0
            else:
                iter_sem_melhora += 1

            self.atualizar_feromonio(melhor_sol)

        tempo_total = time.time() - tempo_inicio
        self.exibir_resultado(melhor_sol, tempo_total)
        self.melhor_sol = melhor_sol
        self.melhor_obj = melhor_sol.obj

    def exibir_resultado(self, sol, tempo):
        print(f"Custo total              : {sol.obj:12.2f}.")
        print(f"Tempo execucao           : {tempo:12.2f} s.")
        #print("itens : valor : peso")
        #for j in range(self.inst.n):
            #if sol.x[j] == 1:
                #print(f"{j:5d} : {self.inst.p[j]:5d} : {self.inst.w[j]:5d}")

class CModel():
    def __init__(self,inst):
        self.inst = inst
        self.create_model()

    def create_model(self):
        inst = self.inst
        N = range(inst.n)
        model = Model('Problema da Mochila',solver_name=CBC)
        x = [model.add_var(var_type=BINARY) for j in N]
        model.objective = maximize(xsum(inst.p[j] * x[j] for j in N))
        model += xsum(inst.w[j] * x[j] for j in N) <= inst.b
        model.verbose = 0
        self.x = x
        self.model = model

    def run(self):
        inst = self.inst
        N = range(inst.n)
        model,x = self.model,self.x
        status = model.optimize()

        if status == OptimizationStatus.OPTIMAL:
            print("\nOptimal solution: {:10.2f}".format(model.objective_value))
            newln = 0
            for j in N:
                if x[j].x > 1e-6:
                    print("{:3d} ".format(j), end='')
                    newln += 1
                    if newln % 10 == 0:
                        newln = 1
                        print()
            print('\n\n')

def main():
    assert len(sys.argv) > 1, 'Uso: python script.py <arquivo_instancia>'
    filename = sys.argv[1]

    resultados = []

    for r in range(100, 3100, 100):
        seed = 2025658375 + r
        np.random.seed(seed)

        # Instancia e executa o ACO
        inst = CInstance(filename)
        aco = CACO(inst)
        start_aco = time.time()
        aco.run()
        end_aco = time.time()
        tempo_aco = end_aco - start_aco
        val_aco = aco.melhor_obj if hasattr(aco, 'melhor_obj') else None

        # Se não houver atributo melhor_obj, pega do melhor_sol
        if val_aco is None and hasattr(aco, 'melhor_sol'):
            val_aco = aco.melhor_sol.obj
        if val_aco is None:
            # Última solução impressa
            val_aco = 0

        # Executa o modelo exato (MIP)
        mod = CModel(inst)
        start_opt = time.time()
        mod.model.optimize()
        end_opt = time.time()
        tempo_opt = end_opt - start_opt

        if mod.model.status == OptimizationStatus.OPTIMAL:
            val_opt = mod.model.objective_value
            gap = 100 * (val_opt - val_aco) / val_opt
            resultados.append([seed, int(val_opt), tempo_opt, int(val_aco), gap, tempo_aco])
        else:
            resultados.append([seed, '-', tempo_opt, int(val_aco), 'N/A', tempo_aco])

    # Impressão da tabela formatada
    print("\n\t\t\tSolução Ótima\t\tACO")
    print("seed\t\tFO\ttime (s)\tFO\terro\ttime (s)")
    for linha in resultados:
        seed, fo_opt, t_opt, fo_aco, erro, t_aco = linha
        fo_opt_str = f"{fo_opt}".ljust(6)
        t_opt_str = f"{t_opt:.2f}".ljust(8) if isinstance(t_opt, float) else f"{t_opt}".ljust(8)
        fo_aco_str = f"{fo_aco}".ljust(6)
        erro_str = f"{erro:.3f}%" if isinstance(erro, float) else erro
        print(f"{seed}\t{fo_opt_str}{t_opt_str}\t{fo_aco_str}{erro_str}\t{t_aco:.2f}")

if __name__ == '__main__':
    main()