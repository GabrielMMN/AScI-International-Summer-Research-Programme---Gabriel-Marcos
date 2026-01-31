import optuna
from aco import CACO, CInstance, CSolution  
import numpy as np

# === CONFIGURAÇÃO DA INSTÂNCIA DE TESTE ===
instancia = CInstance("s010.kp")  

def objective(trial):
    alpha = trial.suggest_float("alpha", 0.5, 5.0)
    beta = trial.suggest_float("beta", 0.5, 5.0)
    rho = trial.suggest_float("rho", 0.01, 0.5)
    Q = trial.suggest_int("Q", 1, 100)
    max_iter = trial.suggest_int("max_iter_sem_melhora", 20, 200)
    tau_init = trial.suggest_float("tau", 0.01, 1.0)

    # Criar uma nova instância do ACO com esses parâmetros
    aco = CACO(instancia, alpha=alpha, beta=beta, rho=rho, Q=Q, max_iter_sem_melhora=max_iter)
    aco.tau = np.full(instancia.n, tau_init)

    melhor_sol = CSolution(instancia)
    iter_sem_melhora = 0
    for _ in range(aco.max_iter_sem_melhora):
        sol = aco.construir_solucao()
        if sol.obj > melhor_sol.obj:
            melhor_sol.copy(sol)
            iter_sem_melhora = 0
        else:
            iter_sem_melhora += 1
        aco.atualizar_feromonio(melhor_sol)
        if iter_sem_melhora >= aco.max_iter_sem_melhora:
            break

    return melhor_sol.obj  # Função objetivo a ser maximizada

# === EXECUÇÃO DA CALIBRAÇÃO ===
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("\nMelhores parâmetros encontrados:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
