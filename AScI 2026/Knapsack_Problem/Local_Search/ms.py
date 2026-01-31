import os
import sys
import numpy as np
from mip import Model, xsum,  maximize, CBC, OptimizationStatus, BINARY
import time
from time import perf_counter as pc

def main():
    assert len(sys.argv) > 1, 'please, provide a data file'
    filename = sys.argv[1]

    resultados = []

    for r in range(100, 3100, 100):
        seed = 2025658375 + r
        np.random.seed(seed)

        # Instancia e solucao
        inst = CInstance(filename)
        sol = CSolution(inst)

        constr = CConstructor()

        # gulosa
        constr.greedy(sol)
        #sol.print()

        ls = CBuscaLocal()

        # Executa Multi-Start
        start_ms = time.time()
        ls.multi_start(sol,5,5,strategy='first')
        end_ms = time.time()
        #sol.print()
        tempo_ms = end_ms - start_ms
        val_ms = sol.obj             # valor da solucao Multi-Start

        # Executa o modelo exato (MIP)
        mod = CModel(inst)
        start_opt = time.time()
        mod.model.optimize()
        end_opt = time.time()
        tempo_opt = end_opt - start_opt

        if mod.model.status == OptimizationStatus.OPTIMAL:
            val_opt = mod.model.objective_value
            gap = 100 * (val_opt - val_ms) / val_opt
            resultados.append([seed, int(val_opt), tempo_opt, int(val_ms), gap, tempo_ms])
        else:
            resultados.append([seed, '-', tempo_opt, int(val_ms), 'N/A', tempo_ms])

    # Impressão da tabela formatada
    print("\n\t\t\tSolução Ótima\t\tMulti-Start")
    print("seed\t\tFO\ttime (s)\tFO\terro\ttime (s)")
    for linha in resultados:
        seed, fo_opt, t_opt, fo_ms, erro, t_ms = linha
        fo_opt_str = f"{fo_opt}".ljust(6)
        t_opt_str = f"{t_opt:.2f}".ljust(8) if isinstance(t_opt, float) else f"{t_opt}".ljust(8)
        fo_ms_str = f"{fo_ms}".ljust(6)
        erro_str = f"{erro:.3f}%" if isinstance(erro, float) else erro
        print(f"{seed}\t{fo_opt_str}{t_opt_str}\t{fo_ms_str}{erro_str}\t{t_ms:.2f}")

class CModel():
    def __init__(self,inst):
        self.inst = inst
        self.create_model()

    def create_model(self):
        inst = self.inst
        N = range(inst.n)
        model = Model('Problema da Mochila',solver_name=CBC)
        # variavel: se j e incluido na mochila
        x = [model.add_var(var_type=BINARY) for j in N]
        # funcao objetivo
        model.objective = maximize(xsum(inst.p[j] * x[j] for j in N))

        # restricao: a capacidade da mochila deve ser respeitada
        model += xsum(inst.w[j] * x[j] for j in N) <= inst.b
        # desliga a impressao do solver
        model.verbose = 0
        self.x = x
        self.model = model

    def run(self):
        inst = self.inst
        N = range(inst.n)
        model,x = self.model,self.x
        status = model.optimize()

        # impressao do resultado
        if status == OptimizationStatus.OPTIMAL:
           print("Optimal solution: {:10.2f}".format(model.objective_value))
           newln = 0
           for j in N:
               if x[j].x > 1e-6:
                   print("{:3d} ".format(j),end='')
                   newln += 1
                   if newln % 10 == 0:
                      newln = 1
                      print()
           print('\n\n')

class CBuscaLocal():
    def __init__(self):
        pass

    def vnd(self, sol, strategy='first'):
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)

        h = 1
        while (h <= 2):
            if strategy == 'first':
                if h == 1:
                    self.swap_one_bit_first_improvement(sol)
                elif h == 2:
                    self.swap_two_bits_first_improvement(sol)
                else:
                    break

            if sol.obj > solstar.obj:
                solstar.copy_solution(sol)
                h = 1
            else:
                h += 1

    def multi_start(self, sol, max_time, max_iterations, strategy='first'):
        crono  = Crono()
        solstar = CSolution(sol.inst)
        solstar.copy_solution(sol)

        h = 0  # contador de iterações

        while h < max_iterations and crono.get_time() < max_time:

            # BuscaLocal
            self.vnd(sol, strategy=strategy)

            # Atualiza melhor solução encontrada, se necessário
            if sol.obj > solstar.obj:
                solstar.copy_solution(sol)

            h += 1

        sol.copy_solution(solstar)  # Retorna a melhor solução
                  
    def swap_one_bit_first_improvement(self, sol):
        inst = sol.inst
        p, w, M = inst.p, inst.w, inst.M
        b, _b = inst.b, sol._b
        N = np.arange(inst.n)
        np.random.shuffle(N)

        for j in N:
            oldval = sol.x[j]
            newval = 0 if oldval else 1

            delta = p[j] * (newval - oldval) \
                + M * max(0, _b - b) \
                - M * max(0, _b + w[j] * (newval - oldval) - b)

            if delta > 0:
                sol.x[j] = newval
                sol.obj += delta
                sol._b += w[j] * (newval - oldval)
                return  # aplica a primeira melhoria e sai

    def swap_two_bits_first_improvement(self, sol):
        inst = sol.inst
        p, w, M = inst.p, inst.w, inst.M
        b, _b = inst.b, sol._b
        n = inst.n
        N = np.arange(n)
        np.random.shuffle(N)

        for h1 in range(n - 1):
            j1 = N[h1]
            oldval1 = sol.x[j1]
            newval1 = 0 if oldval1 else 1

            for h2 in range(h1 + 1, n):
                j2 = N[h2]
                oldval2 = sol.x[j2]
                newval2 = 0 if oldval2 else 1

                delta = p[j1] * (newval1 - oldval1) \
                    + p[j2] * (newval2 - oldval2) \
                    + M * max(0, _b - b) \
                    - M * max(0, _b + w[j1] * (newval1 - oldval1) + w[j2] * (newval2 - oldval2) - b)

                if delta > 0:
                    sol.x[j1] = newval1
                    sol.x[j2] = newval2
                    sol.obj += delta
                    sol._b += w[j1] * (newval1 - oldval1) + w[j2] * (newval2 - oldval2)
                    return  # aplica a primeira melhoria e sai

class CConstructor():
    def __init__(self):
        pass

    def greedy(self,sol):
        inst = sol.inst
        sortedp = inst.p.argsort()[::-1]
        cumsum = np.cumsum(inst.w[sortedp])
        ind = sortedp[np.argwhere(cumsum <= inst.b).ravel()]
        sol.x[:] = 0
        sol.x[ind] = 1 
        sol.z[:] = -1
        sol.z[:len(ind)] = ind[:]
        sol._b = np.sum(inst.w[ind])
        sol.get_obj_val()

class CSolution():
    def __init__(self,inst):
        self.inst = inst
        self.create_structure()

    def create_structure(self):
        self.x = np.zeros(self.inst.n)
        self.z = np.full(self.inst.n,-1)
        self.obj = 0.0
        self._b = self.inst.b

    def get_obj_val(self):
        inst = self.inst
        p,w,b,M = inst.p,inst.w,inst.b,inst.M
        self._b = (self.x * w).sum()
        self.obj = (self.x * p).sum() - M * max(0,self._b-b)
        return self.obj

    def copy_solution(self,sol):
        self.x[:] = sol.x[:]
        self.z[:] = sol.z[:]
        self.obj = sol.obj
        self._b = sol._b

    def print(self):
        self.get_obj_val()
        print(f'obj  : {self.obj:16.2f}')
        print(f'_b/b : {self._b:16.0f}/{self.inst.b:16.0f}')
        newln = 0
        for j,val in enumerate(self.x):
            if val > 0.9:
                print(f'{j:3d} ',end='')
                newln += 1
                if newln % 10 == 0:
                   newln = 1
                   print()
        print('\n\n')

class CInstance():
    def __init__(self,filename):
        self.read_file(filename)

    def read_file(self,filename):
        self.filename = filename
        assert os.path.isfile(filename), 'please, provide a valid file'
        with open(filename,'r') as rf:
            lines = rf.readlines()
            lines = [line for line in lines if line.strip()]
            self.n = int(lines[0])
            self.b = int(lines[1])
            p,w = [],[]
            for h in range(2,self.n+2):
                _p,_w = [int(val) for val in lines[h].split()]
                p.append(_p),w.append(_w)
            self.p,self.w = np.array(p),np.array(w)
        self.M = self.p.sum()

    def print(self):
        print(f'{self.n:9}')
        print(f'{self.b:9}')
        for h in range(self.n):
            print(f'{self.p[h]:4d} {self.w[h]:4d}')

class Crono():
    def __init__(self):
        self.start_time = pc()

    def start(self):
        self.reset()

    def get_time(self):
        return (pc() - self.start_time)

    def reset(self):
        self.start_time = pc()

if __name__ == '__main__':
    main()