import os
import sys
import time
import numpy as np
import numpy.ma as ma
import math
from math import exp,log
from mip import Model, xsum,  maximize, CBC, OptimizationStatus, BINARY

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
    
        # random
        constr.random_solution(sol)
        sol.print()

        ls = CLocalSearch()

        # Executa SA
        start_sa = time.time()
        ls.sa(sol,0.95,10)
        end_sa = time.time()
        tempo_sa = end_sa - start_sa
        val_sa = sol.obj             # valor da solucao SA

        # Executa o modelo exato (MIP)
        mod = CModel(inst)
        start_opt = time.time()
        mod.model.optimize()
        end_opt = time.time()
        tempo_opt = end_opt - start_opt

        if mod.model.status == OptimizationStatus.OPTIMAL:
            val_opt = mod.model.objective_value
            gap = 100 * (val_opt - val_sa) / val_opt
            resultados.append([seed, int(val_opt), tempo_opt, int(val_sa), gap, tempo_sa])
        else:
            resultados.append([seed, '-', tempo_opt, int(val_sa), 'N/A', tempo_sa])

    # Impressão da tabela formatada
    print("\n\t\t\tSolução Ótima\t\tSA")
    print("seed\t\tFO\ttime (s)\tFO\terro\ttime (s)")
    for linha in resultados:
        seed, fo_opt, t_opt, fo_sa, erro, t_sa = linha
        fo_opt_str = f"{fo_opt}".ljust(6)
        t_opt_str = f"{t_opt:.2f}".ljust(8) if isinstance(t_opt, float) else f"{t_opt}".ljust(8)
        fo_sa_str = f"{fo_sa}".ljust(6)
        erro_str = f"{erro:.3f}%" if isinstance(erro, float) else erro
        print(f"{seed}\t{fo_opt_str}{t_opt_str}\t{fo_sa_str}{erro_str}\t{t_sa:.2f}")

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

class CLocalSearch():
    def __init__(self):
        pass

    def swap_bit(self,sol,j):
        inst = sol.inst
        p,w,M,n = inst.p,inst.w,inst.M,inst.n
        b,_b = inst.b,sol._b
        
        oldval,newval = sol.x[j], 0 if sol.x[j] else 1
        delta = p[j] * (newval - oldval)\
              + M * max(0,_b - b)\
              - M * max(0,_b + w[j] * (newval - oldval) - b)
        sol.x[j] = newval 
        sol.obj += delta
        _b += w[j] * (newval - oldval)
        sol._b = _b
        return delta
    
    def sa_initial_temperature(self,sol,alpha,SAmax):
        n = sol.inst.n
        temperature = 2
        print(f'obj : {sol.obj:10.2f}  b : {sol._b:10.2f}')
        accept = 0
        min_accept = int(alpha*SAmax)
        while accept < min_accept: 
           h = 0
           while h < SAmax:
                h += 1
                # choose a random position (item) 
                j = np.random.randint(n)
                delta = self.swap_bit(sol,j) 
                if delta > 0:
                   accept += 1
                else: 
                   rnd = np.random.uniform(0,1)
                   if rnd < exp(delta/temperature):
                      accept += 1
                delta = self.swap_bit(sol,j) 
           if accept < min_accept:
              accept = 0
              temperature  *= 1.1
        print(f'initial temperature:   {temperature:10.2f}')
        return temperature
    
    def sa(self,sol,alpha=0.97,k = 2):
        inst = sol.inst
        p,w,M,n = inst.p,inst.w,inst.M,inst.n
        b,_b = inst.b,sol._b
        N = np.arange(n)
        
        # sa settings
        SAmax = k * n
        initial_temperature = self.sa_initial_temperature(sol,alpha,SAmax)
        temperature = initial_temperature
        final_temperature = 0.01
        n_temp_changes = 0

        # best solution so far
        best_sol = CSolution(inst)
        best_sol.copy(sol)
        while temperature > final_temperature:
            h = 0
            while h < SAmax:
                h += 1
                j = np.random.randint(n)
                delta = self.swap_bit(sol,j) 
                if delta > 0:
                   if sol.obj > best_sol.obj:
                       best_sol.copy(sol)
                       sol.copy(best_sol)                  # copia a melhor solucao encontrada
                else:
                   rnd = np.random.uniform(0,1)
                   if rnd < exp(delta/temperature):
                      pass
                   else:
                      self.swap_bit(sol,j)
            # diminish temperature
            temperature *= alpha
            #print(f'current temperature:   {temperature:10.2f}')
            n_temp_changes += 1
        print(f'final temperature               :{temperature:18.2f}')
        print(f'max number of checked solutions :{n_temp_changes*SAmax:18.0f}') 
        print(f'existing solutions              :{exp(n*log(2)):18.0f}')

class CConstructor():
    def __init__(self):
        pass

    def random_solution(self,sol):
        inst = sol.inst
        N = range(inst.n)
        h = 0
        sol._b = 0
        for j in N:
            val = np.random.choice(2,1)[0]
            sol.x[j] = val
            if val > 0:
               sol._b += inst.w[j]
               sol.z[h] = j
               h += 1
        sol.get_obj_val()

class CSolution():
    def __init__(self,inst):
        self.inst = inst
        self.create_structure()

    def copy_solution(self,sol):
        self.x[:] = sol.x[:]
        self.z[:] = sol.z[:]
        self.obj = sol.obj
        self._b = sol._b

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

    def copy(self,sol):
        self.x[:] =  sol.x[:]
        self.z[:] =  sol.z[:]
        self.obj  =  sol.obj 
        self._b   =  sol._b  

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

    def reset(self):
        self.x[:] =  0
        self.z[:] =  -1
        self.obj  =  0.0
        self._b   =  0.0

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

if __name__ == '__main__':
    main()