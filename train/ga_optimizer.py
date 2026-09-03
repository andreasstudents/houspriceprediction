import random
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from typing import Tuple, Any
from colorama import Fore, Style, init


init(autoreset=True)

logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}%(asctime)s{Style.RESET_ALL} - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ga_optimizer")
from rf_model import train_optimized_rf_model


class GAOptimizer:
    def __init__(self):
        self.initial_population = {"Individu": [], "Parameter": []}
        self.generations_data = {
            "Generasi": [],
            "Best Fitness": [],
            "MAPE": [],
            "Mutation": [],
            "Convergence": [],
        }
        self.evaluation_data = {}
        self.best_individuals = {}
        self.best_overall_fitness = float("-inf")
        self.generations_without_improvement = 0
        self.evaluation_cache = {}

    def save_log(
        self, generation: int, individual_index: int, log_type: str, log_content: Any
    ) -> None:
        """
        Save logging information during GA execution.

        Args:
            generation: Current generation number
            individual_index: Index of individual (-1 for general info)
            log_type: Type of log entry (initial, fitness, crossover, mutation, elitism, final)
            log_content: Data to log
        """
        if log_type == "initial":
            self.initial_population.setdefault("Individu", []).append(
                f"Individu {individual_index + 1}"
            )
            self.initial_population.setdefault("Parameter", []).append(log_content)

        elif log_type == "fitness":
            gen_data = self.evaluation_data.setdefault(
                generation, {"Individu": [], "MAPE (%)": [], "Fitness": []}
            )
            gen_data["Individu"].append(f"Individu {individual_index + 1}")
            gen_data["MAPE (%)"].append(log_content.get("MAPE"))
            gen_data["Fitness"].append(log_content.get("Fitness"))

        elif log_type in ["crossover", "mutation"]:
            gen_data = self.generations_data.setdefault(generation, {})
            gen_data.setdefault(log_type.capitalize(), []).append(log_content)

        elif log_type == "elitism":
            self.best_individuals[generation] = log_content

        elif log_type == "final":
            self.generations_data.setdefault("Generasi", []).append(generation)
            self.generations_data.setdefault("Best Fitness", []).append(
                log_content.get("Best Fitness")
            )
            self.generations_data.setdefault("MAPE", []).append(log_content.get("MAPE"))
            self.generations_data.setdefault("Convergence", []).append(
                log_content.get("Convergence", 0)
            )

    def evaluate_rf_model(
        self,
        params: Tuple,
        data: pd.DataFrame,
        price_target: pd.Series,
        use_cv: bool = True,
        cv_folds: int = 5,
    ) -> float:
        params_key = str(params)
        if params_key in self.evaluation_cache:
            logger.debug(f"Using cached evaluation for params: {params_key}")
            return self.evaluation_cache[params_key]

        try:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            if use_cv:
                # Enhanced cross-validation with multiple metrics
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_scores = []
                r2_scores = []
                rmse_scores = []
                mae_scores = []

                for train_idx, val_idx in kf.split(data):
                    X_train, X_val = data.iloc[train_idx], data.iloc[val_idx]
                    y_train, y_val = (
                        price_target.iloc[train_idx],
                        price_target.iloc[val_idx],
                    )

                    model_tuple = train_optimized_rf_model(X_train, y_train, params)
                    fold_model, _ = model_tuple
                    predictions = fold_model.predict(X_val)
                    
                    # Transform predictions back to original scale for evaluation
                    predictions_original = np.expm1(predictions)
                    y_val_original = np.expm1(y_val)
                    
                    # Calculate metrics on original scale
                    mape = mean_absolute_percentage_error(y_val_original, predictions_original)
                    r2 = r2_score(y_val_original, predictions_original)
                    rmse = np.sqrt(mean_squared_error(y_val_original, predictions_original))
                    mae = mean_absolute_error(y_val_original, predictions_original)
                    
                    cv_scores.append(mape)
                    r2_scores.append(r2)
                    rmse_scores.append(rmse)
                    mae_scores.append(mae)

                # Weighted fitness function combining multiple metrics
                avg_mape = np.mean(cv_scores)
                avg_r2 = np.mean(r2_scores)
                avg_rmse = np.mean(rmse_scores)
                avg_mae = np.mean(mae_scores)
                
                # Multi-objective fitness (lower is better for GA maximization)
                # Weights: MAPE (40%), R2 (30%), RMSE (20%), MAE (10%)
                normalized_rmse = min(avg_rmse / 1e9, 1.0)  # Normalize RMSE
                normalized_mae = min(avg_mae / 1e9, 1.0)   # Normalize MAE
                
                composite_score = (
                    0.4 * avg_mape +
                    0.3 * (1 - max(avg_r2, 0)) +  # Convert R2 to loss (lower is better)
                    0.2 * normalized_rmse +
                    0.1 * normalized_mae
                )
                
                result = composite_score
            else:
                model_tuple = train_optimized_rf_model(data, price_target, params)
                model, _ = model_tuple
                predictions = model.predict(data)
                
                # Transform predictions back to original scale
                predictions_original = np.expm1(predictions)
                price_target_original = np.expm1(price_target)
                
                result = mean_absolute_percentage_error(price_target_original, predictions_original)

            self.evaluation_cache[params_key] = result
            return result
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error evaluating RF model: {str(e)}{Style.RESET_ALL}"
            )
            return 1.0

    def fitness_function(
        self,
        params: Tuple,
        data: pd.DataFrame,
        price_target: pd.Series,
        cv_folds: int = 5,
    ) -> float:
        mape = self.evaluate_rf_model(params, data, price_target, cv_folds=cv_folds)
        return -mape

    def create_initial_population(self, size, object_bounds):
        population = []
        for _ in range(size):
            individual = tuple(
                random.uniform(lower_bound, upper_bound)
                for lower_bound, upper_bound in object_bounds
            )
            population.append(individual)
        return population

    def selection(self, population, fitnesses, tournament_size=3):
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(
                list(zip(population, fitnesses)), tournament_size
            )
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        alpha = random.random()
        child1 = tuple(
            alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)
        )
        child2 = tuple(
            alpha * p2 + (1 - alpha) * p1 for p1, p2 in zip(parent1, parent2)
        )
        return child1, child2

    def mutation(
        self, individual, mutation_rate, object_bounds, generation=0, max_generations=10
    ):
        individual = list(individual)

        if max_generations > 1:
            adaptive_rate = mutation_rate * (1 - (generation / max_generations))
        else:
            adaptive_rate = mutation_rate

        for i in range(len(individual)):

            if random.random() < adaptive_rate:
                lower_bound, upper_bound = object_bounds[i]

                sigma = (upper_bound - lower_bound) * 0.1
                mutation_amount = random.gauss(0, sigma)

                individual[i] += mutation_amount
                individual[i] = max(min(individual[i], upper_bound), lower_bound)

        return tuple(individual)

    def genetic_algorithm(
        self,
        population_size,
        object_bounds,
        generations,
        mutation_rate,
        data,
        price_target,
        early_stopping_generations=5,
        convergence_threshold=0.001,
        cv_folds=3,
    ):
        population = self.create_initial_population(population_size, object_bounds)

        for i, ind in enumerate(population):
            self.save_log(
                generation=0, individual_index=i, log_type="initial", log_content=ind
            )

        best_performers = []
        self.best_overall_fitness = float("-inf")
        self.generations_without_improvement = 0
        previous_best_fitness = float("-inf")

        for generation in range(1, generations + 1):
            logger.info(
                f"{Fore.MAGENTA}Generation {generation}/{generations}{Style.RESET_ALL}"
            )

            fitnesses = [
                self.fitness_function(
                    ind, data, price_target=price_target, cv_folds=cv_folds
                )
                for ind in population
            ]
            for i, (ind, fitness) in enumerate(zip(population, fitnesses)):
                log_content = {"MAPE": -fitness, "Fitness": fitness}
                self.save_log(generation, i, "fitness", log_content)

            best_index = fitnesses.index(max(fitnesses))
            best_individual = population[best_index]
            best_fitness = fitnesses[best_index]
            best_mape = -best_fitness
            best_performers.append((best_individual, best_fitness))

            if best_fitness > self.best_overall_fitness:
                improvement = best_fitness - self.best_overall_fitness
                self.best_overall_fitness = best_fitness
                self.generations_without_improvement = 0
                convergence_metric = improvement / (
                    abs(self.best_overall_fitness) + 1e-10
                )
                logger.info(
                    f"{Fore.GREEN}Gen {generation}: Improved fitness by {improvement:.6f} (MAPE: {-best_fitness*100:.2f}%, convergence: {convergence_metric:.6f}){Style.RESET_ALL}"
                )
            else:
                self.generations_without_improvement += 1
                convergence_metric = 0
                logger.info(
                    f"{Fore.YELLOW}Gen {generation}: No improvement for {self.generations_without_improvement} generations (Current MAPE: {-best_fitness*100:.2f}%){Style.RESET_ALL}"
                )

            if previous_best_fitness != float("-inf"):
                relative_improvement = (best_fitness - previous_best_fitness) / (
                    abs(previous_best_fitness) + 1e-10
                )
            else:
                relative_improvement = 1.0

            previous_best_fitness = best_fitness

            self.generations_data["Generasi"].append(generation)
            self.generations_data["Best Fitness"].append(best_fitness)
            self.generations_data["MAPE"].append(best_mape)
            self.generations_data["Convergence"].append(convergence_metric)

            best_individual_data = {
                "Parameter": best_individual,
                "Fitness": best_fitness,
            }
            self.save_log(generation, best_index, "elitism", best_individual_data)

            selected_population = self.selection(population, fitnesses)

            next_population = []

            for i in range(0, len(selected_population) - 1, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                crossover_info = {
                    "Parent1": parent1,
                    "Parent2": parent2,
                    "Child1": child1,
                    "Child2": child2,
                }
                self.save_log(generation, i, "crossover", crossover_info)
                next_population.extend([child1, child2])

            if len(selected_population) % 2 != 0:
                next_population.append(selected_population[-1])
                self.save_log(
                    generation,
                    -1,
                    "crossover",
                    f"Odd individual carried to next generation: {selected_population[-1]}",
                )

            mutated_population = []

            current_mutation_rate = mutation_rate * (
                1 - 0.5 * (generation / generations)
            )
            self.generations_data["Mutation"].append(current_mutation_rate)

            for i, ind in enumerate(next_population):
                mutated = self.mutation(
                    ind, current_mutation_rate, object_bounds, generation, generations
                )
                mutation_info = {"Original": ind, "Mutated": mutated}
                self.save_log(generation, i, "mutation", mutation_info)
                mutated_population.append(mutated)

            mutated_population[0] = best_individual
            self.save_log(generation, 0, "elitism", best_individual)

            population = mutated_population

            if self.generations_without_improvement >= early_stopping_generations:
                logger.info(
                    f"{Fore.YELLOW}Early stopping at generation {generation} (No improvement for {early_stopping_generations} generations){Style.RESET_ALL}"
                )
                break

            if (
                convergence_metric < convergence_threshold
                and generation > early_stopping_generations
            ):
                logger.info(
                    f"{Fore.GREEN}Converged at generation {generation} (metric: {convergence_metric:.6f} < threshold: {convergence_threshold}){Style.RESET_ALL}"
                )
                break

        final_best_individual = max(best_performers, key=lambda x: x[1])[0]
        final_best_fitness = max(best_performers, key=lambda x: x[1])[1]
        final_best_mape = -final_best_fitness
        final_results = {
            "Best Individual": final_best_individual,
            "Fitness": final_best_fitness,
            "MAPE": final_best_mape,
            "Convergence": self.generations_without_improvement,
            "Cache Hits": len(self.evaluation_cache),
        }
        self.save_log(generations, -1, "final", final_results)

        logger.info(
            f"{Fore.GREEN}Optimization complete - Best MAPE: {final_best_mape*100:.2f}% | Cached evaluations: {len(self.evaluation_cache)}{Style.RESET_ALL}"
        )

        return final_best_individual, best_performers

    def save_logs_to_json(self, file_path):
        data = {
            "initial_population": self.initial_population,
            "evaluation_data": self.evaluation_data,
            "generations_data": self.generations_data,
            "best_individuals": self.best_individuals,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
