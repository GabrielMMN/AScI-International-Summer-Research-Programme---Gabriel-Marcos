import os
import sys
import numpy as np
import numpy.ma as ma
import math
from heapq import heapify,heappush,merge
from math import exp,log
from mip import Model, xsum,  maximize, CBC, OptimizationStatus, BINARY
import time


class CModel():
    def __init__(self,inst):
        self.inst = inst
        self.create_model()

    def create_model(self):
        inst = self.inst
        N = range(inst.n)
        model = Model('Problema da Mochila',solver_name=CBC)
        # variavel: se o projeto j e incluido na mochila
        x = [model.add_var(var_type=BINARY) for j in N]
        # funcao objetivo: maximizar o retorno
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
        # otimiza o modelo chamando o resolvedor
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

    def brkga(self,inst,population_size=50,\
                   elite_population_size=25,\
                   mutant_population_size=10,\
                   elite_inheritance_prob=90,\
                   max_generation=200):

        def combine_solution(offspring,pa,pb):
            randvals =  np.random.rand(pa.inst.n)
            is_xor =  randvals < (elite_inheritance_prob/100.0)
            offspring.x = np.where(is_xor, pa.x, pb.x) 
            offspring.get_obj_val()

        if population_size % 2 != 0:
           print('population size must be even. Population will be increased')
           population_size += 1

        P = range(population_size) 
        #initial population
        pop = [CSolution(inst) for p in range(2 * population_size)]
        constr = CConstructor()
        for p in pop:
            constr.brkga_initial_solution(p)
            obj = p.get_obj_val()

        endelitepop = int(population_size * (elite_population_size /100))
        beginmutantpop = population_size -  int(population_size * (mutant_population_size /100))

        #is_literature = 0
        #is_literature = 1
        is_literature = 2
        if is_literature == 0:
           pop.sort(key=lambda p : p.obj,reverse=True)
           # main loop
           ngs = 0
           while ngs < max_generation:
               ngs += 1
          
               # copy elite group to the future generation 
               h = 0
               for p in pop[population_size:population_size+endelitepop]:
                   p.copy(pop[h])
                   h += 1
          
               # introduce the mutants at the end of population 
               for p in pop[beginmutantpop:population_size]:
                   constr.brkga_initial_solution(p)
                   obj = p.get_obj_val()
           
               # new ofsprings
               noffsprings = population_size + endelitepop
               while noffsprings < 2 * population_size: 
                    idx_father_a = np.random.randint(0,endelitepop,size=1)[0]
                    idx_father_b = np.random.randint(endelitepop,population_size,size=1)[0]
                    combine_solution(pop[noffsprings],pop[idx_father_a],pop[idx_father_b]) 
                    noffsprings += 1
                
               # sort to create new elite group 
               pop.sort(key=lambda p : p.obj,reverse=True)
           
               #print(f'{ngs:4d} {pop[0].obj:12.2f}') 
        elif is_literature == 1:
            pop.sort(key=lambda p : p.obj,reverse=True)
            # main loop
            ngs = 0
            while ngs < max_generation:
                ngs += 1
            
                # new ofsprings
                noffsprings = population_size 
                while noffsprings < 2 * population_size: 
                     idx_father_a = np.random.randint(0,endelitepop,size=1)[0]
                     idx_father_b = np.random.randint(endelitepop,population_size,size=1)[0]
                     combine_solution(pop[noffsprings],pop[idx_father_a],pop[idx_father_b]) 
                     noffsprings += 1
            
                pop.sort(key=lambda p : p.obj,reverse=True)

                for p in pop[beginmutantpop:population_size]:
                    constr.brkga_initial_solution(p)
                    obj = p.get_obj_val()
            
                # print(f'{ngs:4d} {pop[0].obj:12.2f}') 
        elif is_literature == 2: 
            pop.sort(key=lambda p : p.obj,reverse=True)
            # main loop
            ngs = 0
            while ngs < max_generation:
                ngs += 1
            
                # new ofsprings
                noffsprings = population_size 
                while noffsprings < 2 * population_size: 
                     #idx_father_a = np.random.randint(0,endelitepop,size=1)[0]
                     idx_father_a = np.random.randint(0,population_size,size=1)[0]
                     idx_father_b = np.random.randint(0,population_size,size=1)[0]
                     combine_solution(pop[noffsprings],pop[idx_father_a],pop[idx_father_b]) 
                     noffsprings += 1
            
                pop.sort(key=lambda p : p.obj,reverse=True)

                for p in pop[beginmutantpop:population_size]:
                    constr.brkga_initial_solution(p)
                    obj = p.get_obj_val()
            
                #print(f'{ngs:4d} {pop[0].obj:12.2f}') 
 
        print(f'best individual {pop[0].obj:12.2f}')

    def ga(self,inst,population_size=20,max_generation=10,prob_xor=0.50,prob_mutation=.10, best_sol=None):  # <-- Adicionado best_sol
        def repair_solution(sol):
           if sol.obj < 0:
              self.swap_one_bit(sol)
           while sol.obj < 0:
               idx = np.asarray(sol.x == 1).nonzero()[0]
               j = np.random.choice(idx,1)[0]
               self.swap_bit(sol,j)

        def roulette(population,population_size,prob):
            totalobj = sum([p.obj for p in population])
            for j,p in enumerate(population):
                prob[j] = p.obj /totalobj
            idx = np.random.choice(len(population),population_size,replace=False,p=prob)
            idx.sort()
            h = 0
            while h < population_size:
                s = idx[h]
                population[h].copy(population[s])
                h += 1

        if population_size % 2 != 0:
           print('population size must be even. Population will be increased')
           population_size += 1
      
        nchildren = int(population_size/2) if int( population_size/2 ) % 2 == 0 else int(population_size/2) + 1

        P = [CSolution(inst) for p in range(population_size + nchildren)]

        prob = np.zeros(len(P)) 

        constr = CConstructor()

        best_sol_local = CSolution(inst)  # <-- Renomeado para evitar conflito
        constr.partial_greedy(best_sol_local,alpha=.10)
         
        for h,p in enumerate(P[:population_size]):
            constr.partial_greedy(p,alpha=.10)
            repair_solution(p)
            if p.obj > best_sol_local.obj:
               best_sol_local.copy(p)

        ngenerations = 0
        while ngenerations < max_generation:
            ngenerations += 1
            h = 0
            while h < nchildren:
               j1,j2 = np.random.choice(population_size,2,replace=False)
               if np.random.uniform(0,1) < prob_xor:
                  point = np.random.randint(1,inst.n-2)
                  # crossover
                  P[population_size + h].copy(P[j1]) 
                  P[population_size + h + 1].copy(P[j2])
                  P[population_size + h].x[point:] = P[j2].x[point:]
                  P[population_size + h].get_obj_val()
                  P[population_size + h + 1].x[point:] = P[j1].x[point:]
                  P[population_size + h + 1].get_obj_val()
                  # mutation
                  if np.random.uniform(0,1) < prob_mutation:
                     j = np.random.randint(inst.n)
                     self.swap_bit(P[population_size + h],j)
                  if np.random.uniform(0,1) < prob_mutation:
                     j = np.random.randint(inst.n)
                     self.swap_bit(P[population_size + h + 1],j)
                  # repair solution if infeasible 
                  if P[population_size + h].obj < 0:
                     repair_solution(P[population_size + h])
                  if P[population_size + h + 1].obj < 0:
                     repair_solution(P[population_size + h + 1])
                  # update best solution
                  if P[population_size + h].obj > best_sol_local.obj:
                    self.swap_one_bit(P[population_size + h])
                    best_sol_local.copy(P[population_size + h])
                  if P[population_size + h+1].obj > best_sol_local.obj:
                    self.swap_one_bit(P[population_size + h + 1])
                    best_sol_local.copy(P[population_size + h + 1])
                  h += 2
            roulette(P,population_size,prob)
 
        # Copia a melhor solução encontrada para o argumento best_sol, se fornecido
        if best_sol is not None:
            best_sol.copy(best_sol_local)
        # best_sol_local.print()  # Remova ou comente para não poluir a saída

    def swap_one_bit(self,sol):
        inst = sol.inst
        p,w,M = inst.p,inst.w,inst.M
        b,_b = inst.b,sol._b
        N = np.arange(inst.n)

        best_delta = float('inf')
        best_j = -1

        while best_delta > 0:
              np.random.shuffle(N)
              best_delta = -float('inf')

              for j in N:
                  oldval,newval = sol.x[j], 0 if sol.x[j] else 1
                  delta = p[j] * (newval - oldval)\
                        + M * max(0,_b - b)\
                        - M * max(0,_b + w[j] * (newval - oldval) - b)
                  if delta > 0:
                      best_j = j
                      best_delta = delta
                      oldval,newval = sol.x[best_j], 0 if sol.x[best_j] else 1
                      sol.x[best_j] = newval 
                      sol.obj += best_delta
                      _b += w[best_j] * (newval - oldval)
                      sol._b = _b
                      

class CConstructor():
    def __init__(self):
        pass

    def brkga_initial_solution(self,sol):
        sol.x[:] = np.random.rand(sol.inst.n)

    def random_solution2(self,sol):
        inst = sol.inst
        p = np.random.choice(inst.n,1)[0]
        vals = np.random.choice(inst.n,p,replace=False)
        sol.x[:] = 0
        sol.z[:] = -1
        sol.x[vals] = 1
        sol.z[:p] = vals[:]
        sol._b = inst.w[vals].sum()
        sol.get_obj_val()

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
    
    def partial_greedy(self,sol,alpha):
        inst = sol.inst
        sol.reset()

        N = range(inst.n)

        stop = False
        ls = CLocalSearch()

        rb = np.zeros(inst.n)

        while stop == False:

            for j in N:
                if sol.x[j] == False:
                    delta = ls.swap_bit(sol,j)
                    rb[j] = sol.obj 
                    delta = ls.swap_bit(sol,j)

            masked = ma.masked_array(rb,mask=sol.x)
            maxrb = masked.max()
            minrb = masked.min()
            interval = maxrb - alpha * (maxrb - minrb)

            items = ma.where(masked >= interval)[0]

            if len(items) > 0 and maxrb > 1e-6:
               j = np.random.choice(items,1)[0]
               ls.swap_bit(sol,j)
               if sol.obj < 1e-6:
                  ls.swap_bit(sol,j)
                  stop = True
            else: 
                stop = True


class CSolution():
    def __init__(self,inst):
        self.inst = inst
        self.create_structure()
    
    def __lt__(self, other):
        return self.obj > other.obj

    def create_structure(self):
        self.x = np.zeros(self.inst.n)
        self.z = np.full(self.inst.n,-1)
        self.obj = 0.0
        self._b = self.inst.b

    def get_obj_val(self):
        inst = self.inst
        n,p,w,b,M = inst.n,inst.p,inst.w,inst.b,inst.M
        val = (self.x > 1e-6) & (self.x < 1-1e-6)
        if np.any(val,where = True):
           _x = np.ndarray.flatten(np.argwhere(self.x > 0.5))
           x = np.zeros(n)
           x[_x] = 1.0
        else:
           x = self.x
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

def main():
    assert len(sys.argv) > 1, 'please, provide a data file'
    filename = sys.argv[1]

    resultados = []

    for r in range(100, 3100, 100):  # Agora cobre 30 seeds
        seed = 2025658375 + r
        np.random.seed(seed)

        # Instancia e executa o GA
        inst = CInstance(filename)
        ls = CLocalSearch()
        start_ga = time.time()
        best_sol = CSolution(inst)
        ls.ga(inst,
              population_size=inst.n * 2,
              max_generation=200,
              prob_xor=0.90,
              prob_mutation=0.2,
              best_sol=best_sol)
        end_ga = time.time()
        tempo_ga = end_ga - start_ga
        val_ga = best_sol.obj

        # Executa o modelo exato (MIP)
        mod = CModel(inst)
        start_opt = time.time()
        mod.model.optimize()
        end_opt = time.time()
        tempo_opt = end_opt - start_opt

        if mod.model.status == OptimizationStatus.OPTIMAL:
            val_opt = mod.model.objective_value
            gap = 100 * (val_opt - val_ga) / val_opt
            resultados.append([seed, int(val_opt), tempo_opt, int(val_ga), gap, tempo_ga])
        else:
            resultados.append([seed, '-', tempo_opt, int(val_ga), 'N/A', tempo_ga])

    # Impressão da tabela formatada
    print("\n\t\t\tSolução Ótima\t\tGA")
    print("seed\t\tFO\ttime (s)\tFO\terro\ttime (s)")
    for linha in resultados:
        seed, fo_opt, t_opt, fo_ga, erro, t_ga = linha
        fo_opt_str = f"{fo_opt}".ljust(6)
        t_opt_str = f"{t_opt:.2f}".ljust(8) if isinstance(t_opt, float) else f"{t_opt}".ljust(8)
        fo_ga_str = f"{fo_ga}".ljust(6)
        erro_str = f"{erro:.3f}%" if isinstance(erro, float) else erro
        print(f"{seed}\t{fo_opt_str}{t_opt_str}\t{fo_ga_str}{erro_str}\t{t_ga:.2f}")

if __name__ == '__main__':
    main()