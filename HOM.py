import numpy as np
import pandas as pd
import optuna
from abc import ABC, abstractmethod
from tqdm import tqdm
from numba import njit
from typing import Tuple
import time 


STOPPING_CRITERIA = 12
MAX_ITERATIONS = 30
SKIP_SHEET = [
    "VFR100_20_10",
    "VFR100_40_10",
    "VFR100_60_10",
    "VFR200_20_10",
    "VFR200_40_10",
    "VFR200_60_10",
    "VFR400_20_10",
    "VFR400_40_10"
    # "VRF400_60_10",
    # "VRF600_20_10"
    ]

@njit
def calculate_makespan(jobs_sequence: np.ndarray, processing_times: np.ndarray) -> int:
    num_jobs, num_machines = len(jobs_sequence), processing_times.shape[1]
    completion_time = np.zeros((num_jobs, num_machines), dtype=np.int32)

    for i in range(num_jobs):
        job_index = jobs_sequence[i]
        for j in range(num_machines):
            if i == 0 and j == 0:
                completion_time[i, j] = processing_times[job_index, j]
            elif i == 0:
                completion_time[i, j] = completion_time[i, j - 1] + processing_times[job_index, j]
            elif j == 0:
                completion_time[i, j] = completion_time[i - 1, j] + processing_times[job_index, j]
            else:
                completion_time[i, j] = max(completion_time[i - 1, j], completion_time[i, j - 1]) + processing_times[job_index, j]

    return int(completion_time[num_jobs - 1, num_machines - 1])


@njit
def neh_heuristic(processing_times: np.ndarray) -> np.ndarray:
    num_jobs = processing_times.shape[0]
    # Step 1: Calculate the total processing time for each job
    total_processing_times = np.sum(processing_times, axis=1)
    # Step 2: Sort jobs in descending order of total processing time
    sorted_jobs = np.argsort(-total_processing_times)
    # Step 3: Build the sequence iteratively
    partial_sequence = np.empty(1, dtype=np.int64)
    partial_sequence[0] = sorted_jobs[0]

    for i in range(1, num_jobs):
        best_sequence = None
        best_makespan = float('inf')
        current_job = sorted_jobs[i]
        # Try inserting the current job in all possible positions
        for position in range(len(partial_sequence) + 1):
            new_sequence = np.empty(len(partial_sequence) + 1, dtype=np.int64)
            new_sequence[:position] = partial_sequence[:position]
            new_sequence[position] = current_job
            new_sequence[position + 1:] = partial_sequence[position:]
            makespan = calculate_makespan(new_sequence, processing_times)
            if makespan < best_makespan:
                best_sequence = new_sequence
                best_makespan = makespan
        partial_sequence = best_sequence

    return partial_sequence


@njit
def two_opt_operator(jobs_sequence: np.ndarray, processing_times: np.ndarray) -> Tuple[np.ndarray, int]:
    best_sequence = jobs_sequence.copy()
    best_makespan = calculate_makespan(best_sequence, processing_times)
    n = len(jobs_sequence)
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            new_sequence = best_sequence.copy()
            # Swap two jobs to create a new sequence
            new_sequence[i:j+1] = new_sequence[i:j+1][::-1]
            new_makespan = calculate_makespan(new_sequence, processing_times)
            if new_makespan < best_makespan:
                best_sequence = new_sequence
                best_makespan = new_makespan
                
    return best_sequence, best_makespan


@njit
def greedy_insertion_operator(jobs_sequence: np.ndarray, processing_times: np.ndarray) -> Tuple[np.ndarray, int]:
    best_sequence = jobs_sequence.copy()
    best_makespan = calculate_makespan(best_sequence, processing_times)
    n = len(jobs_sequence)

    for i in range(n):
        # Create a new list that excludes the current job at index i
        temp_sequence = [best_sequence[k] for k in range(n) if k != i]

        current_job = best_sequence[i]

        for j in range(n):
            # Create a new sequence with current_job inserted at position j
            new_sequence = temp_sequence[:j] + [current_job] + temp_sequence[j:]
            new_sequence = np.array(new_sequence)  # Convert back to numpy array for calculation

            new_makespan = calculate_makespan(new_sequence, processing_times)
            if new_makespan < best_makespan:
                best_sequence = new_sequence
                best_makespan = new_makespan

    return best_sequence, best_makespan


@njit
def randomized_descent_operator(jobs_sequence: np.ndarray, processing_times: np.ndarray, k: int = 5) -> Tuple[np.ndarray, int]:
    best_sequence = jobs_sequence.copy()
    best_makespan = calculate_makespan(best_sequence, processing_times)
    n = len(jobs_sequence)
    
    # Divide the sequence into k blocks
    block_size = n // k
    blocks = [jobs_sequence[i * block_size: (i + 1) * block_size] for i in range(k)]
    
    # Handle any remainder elements by adding them to the last block
    remainder = n % k
    if remainder > 0:
        remainder_elements = jobs_sequence[-remainder:]
        last_block = blocks[-1]
        new_last_block = np.empty(len(last_block) + remainder, dtype=jobs_sequence.dtype)
        
        # Copy elements from the last block
        for i in range(len(last_block)):
            new_last_block[i] = last_block[i]
        # Copy remainder elements
        for i in range(remainder):
            new_last_block[len(last_block) + i] = remainder_elements[i]
        
        blocks[-1] = new_last_block
    
    # Randomly permute the indices within each block
    permuted_blocks = []
    for block in blocks:
        permuted_block = block.copy()
        np.random.shuffle(permuted_block)
        permuted_blocks.append(permuted_block)
    
    # Stitch the blocks together in the same order of the original blocks
    new_sequence = np.empty(n, dtype=jobs_sequence.dtype)
    index = 0
    for block in permuted_blocks:
        for i in range(len(block)):
            new_sequence[index] = block[i]
            index += 1
    
    new_makespan = calculate_makespan(new_sequence, processing_times)
    
    if new_makespan < best_makespan:
        best_sequence = new_sequence
        best_makespan = new_makespan
    
    return best_sequence, best_makespan


class Process:
    def __init__(self, jobs_sequence: np.ndarray, processing_times: np.ndarray, verbose: bool = False):
        self.jobs_sequence = jobs_sequence
        self.processing_times = processing_times
        self.verbose = verbose

    def Run(self, meta_cls, **kwargs):
        metaheuristic = meta_cls(self.jobs_sequence, self.processing_times, **kwargs)
        result = metaheuristic.run()
        if self.verbose:
            print(f"{meta_cls.__name__} completed with best makespan: {result[1]}")
        return result


class Metaheuristic(ABC):
    def __init__(self, jobs_sequence: np.ndarray, processing_times: np.ndarray):
        self.jobs_sequence = jobs_sequence.copy()
        self.processing_times = processing_times
        self.best_sequence = self.jobs_sequence.copy()
        self.best_makespan = calculate_makespan(self.jobs_sequence, self.processing_times)
    
    @abstractmethod
    def run(self) -> Tuple[np.ndarray, int]:
        pass


class QLearning(Metaheuristic):
    def __init__(self, jobs_sequence: np.ndarray, processing_times: np.ndarray, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(jobs_sequence, processing_times)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = None

    def run(self) -> Tuple[np.ndarray, int]:
        self.q_table = np.zeros((10, 3))
        return self._q_learning(self.jobs_sequence, self.processing_times, self.q_table, self.alpha, self.gamma, self.epsilon)
    
    @staticmethod
    @njit
    def _q_learning(
        jobs_sequence: np.ndarray, 
        processing_times: np.ndarray, 
        q_table: np.ndarray, alpha: float, 
        gamma: float, 
        epsilon: float
        ) -> Tuple[np.ndarray, int]:

        current_sequence = jobs_sequence.copy()
        best_sequence = current_sequence.copy()
        best_makespan = calculate_makespan(current_sequence, processing_times)
        no_improvement_counter = 0

        episode = 0
        while (no_improvement_counter < STOPPING_CRITERIA // 6 or episode < MAX_ITERATIONS // 6):
            current_state = episode % len(q_table)

            # Choose an operator randomly or exploit based on epsilon
            if np.random.random() < epsilon:
                operator_type = np.random.randint(0, 3)
            else:
                for _ in range(6):
                    greedy_sequence, greedy_makespan = greedy_insertion_operator(current_sequence, processing_times)
                    two_opt_sequence, two_opt_makespan = two_opt_operator(current_sequence, processing_times)
                    random_descent_sequence, random_makespan = randomized_descent_operator(current_sequence, processing_times)
                operator_type = np.argmin(np.array([greedy_makespan, two_opt_makespan, random_makespan]))

            if operator_type == 0:
                for _ in range(6):
                    greedy_sequence, greedy_makespan = greedy_insertion_operator(current_sequence, processing_times)
                new_sequence, new_makespan = greedy_sequence, greedy_makespan
            elif operator_type == 1:
                for _ in range(6):
                    two_opt_sequence, two_opt_makespan = two_opt_operator(current_sequence, processing_times)
                new_sequence, new_makespan = two_opt_sequence, two_opt_makespan  
            else:
                for _ in range(6):
                    random_descent_sequence, random_makespan = randomized_descent_operator(current_sequence, processing_times)
                new_sequence, new_makespan = random_descent_sequence, random_makespan

            # Update best solution if improved
            if new_makespan < best_makespan:
                best_sequence = new_sequence
                best_makespan = new_makespan
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            # Update current sequence for next iteration
            current_sequence = new_sequence

            # Update Q-table
            next_state = (episode + 1) % len(q_table)
            reward = best_makespan - new_makespan
            q_table[current_state, operator_type] += alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[current_state, operator_type]
                )

            episode += 1

        return best_sequence, best_makespan, episode * 6
    

class RandomSelection(Metaheuristic):
    def __init__(self, jobs_sequence: np.ndarray, processing_times: np.ndarray):
        super().__init__(jobs_sequence, processing_times)

    def run(self) -> Tuple[np.ndarray, int]:
        return self._random_selection(self.jobs_sequence, self.processing_times)
    
    @staticmethod
    @njit
    def _random_selection(jobs_sequence: np.ndarray, processing_times: np.ndarray) -> Tuple[np.ndarray, int]:
        current_sequence = jobs_sequence.copy()
        best_sequence = current_sequence.copy()
        best_makespan = calculate_makespan(current_sequence, processing_times)
        no_improvement_counter = 0
        i = 0

        while (no_improvement_counter < STOPPING_CRITERIA or i < MAX_ITERATIONS):
            operator_index = np.random.randint(0, 3)
            if operator_index == 0:
                new_sequence, new_makespan = greedy_insertion_operator(current_sequence, processing_times)
            elif operator_index == 1:
                new_sequence, new_makespan = randomized_descent_operator(current_sequence, processing_times)
            else:
                new_sequence, new_makespan = two_opt_operator(current_sequence, processing_times)

            if new_makespan < best_makespan:
                best_makespan = new_makespan
                best_sequence = new_sequence
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            current_sequence = new_sequence
            i += 1

        return best_sequence, best_makespan, i


def objective(trial, seed: int, jobs_sequence: np.ndarray, processing_times: np.ndarray) -> int:
    alpha = trial.suggest_float("alpha", 0.01, 1.0, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 0.99)
    epsilon = trial.suggest_float("epsilon", 0.01, 1.0)

    np.random.seed(seed)

    q_learning_metaheuristic = QLearning(jobs_sequence, processing_times, alpha, gamma, epsilon)
    _, best_makespan, _ = q_learning_metaheuristic.run()

    return best_makespan

def run_optimization(file_path: str, seed: int):
    excel_data = pd.read_excel(file_path, sheet_name=None)
    first_sheet_name = list(excel_data.keys())[0]
    sheet_data = excel_data[first_sheet_name]

    processing_times = sheet_data.values
    initial_sequence = neh_heuristic(processing_times)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, seed, initial_sequence, processing_times), n_trials=25, n_jobs=-1)

    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    return best_params

def main(file_path: str, seeds: list, best_params: dict, cold_start: bool = False) -> pd.DataFrame:
    results = []
    excel_data = pd.read_excel(file_path, sheet_name=None)

    total_iterations = len(excel_data) * len(seeds)
    pbar = tqdm(total=total_iterations, desc="Processing Sheets and Seeds")

    for sheet_name, sheet_data in excel_data.items():
        processing_times = sheet_data.values

        if sheet_name in SKIP_SHEET:
            pbar.update(len(seeds))
            continue
        
        if cold_start:
            initial_sequence = np.arange(sheet_data.shape[0])
        else:
            initial_sequence = neh_heuristic(processing_times)

        
        initial_makespan = calculate_makespan(initial_sequence, processing_times)
        print(f"Initial Sequence for {sheet_name} done.")

        for seed in seeds:
            np.random.seed(seed)
            process = Process(initial_sequence, processing_times)

            start_time = time.time()
            random_selection_result = process.Run(RandomSelection)
            end_time_random = time.time() - start_time

            start_time = time.time()
            q_learning_result = process.Run(QLearning, **best_params)
            end_time_q = time.time() - start_time

            if cold_start:
                results.extend([
                    {"Sheet Name": sheet_name, 
                    "Metaheuristic": "RandomSelection", 
                    "Seed": seed, 
                    "Best Sequence": random_selection_result[0], 
                    "Best Makespan": random_selection_result[1],
                    "Iterations": random_selection_result[2],
                    "Time": end_time_random},
                    {"Sheet Name": sheet_name,
                    "Metaheuristic": "QLearning", 
                    "Seed": seed, 
                    "Best Sequence": q_learning_result[0], 
                    "Best Makespan": q_learning_result[1],
                    "Iterations": q_learning_result[2],
                    "Time": end_time_q}
                ])
            else:
                results.extend([
                    {"Sheet Name": sheet_name, 
                    "Metaheuristic": "RandomSelection", 
                    "Seed": seed, 
                    "Initial Sequence": initial_sequence,
                    "Initial Makespan": initial_makespan,
                    "Best Sequence": random_selection_result[0], 
                    "Best Makespan": random_selection_result[1],
                    "Iterations": random_selection_result[2],
                    "Time": end_time_random},
                    {"Sheet Name": sheet_name,
                    "Metaheuristic": "QLearning", 
                    "Seed": seed, 
                    "Initial Sequence": initial_sequence,
                    "Initial Makespan": initial_makespan,
                    "Best Sequence": q_learning_result[0], 
                    "Best Makespan": q_learning_result[1],
                    "Iterations": q_learning_result[2],
                    "Time": end_time_q}
                ])

            pbar.update(1)
        
        temp_results = pd.DataFrame(results)
        temp_results.to_csv("Temp6.csv", index=False)

    pbar.close()
    result_table = pd.DataFrame(results)
    print("Processing completed!")
    return result_table

if __name__ == "__main__":
    file_path = "Instances.xlsx"
    seeds = [0, 42, 93, 7043, 101112]
    best_params = {'alpha': 0.27120573969121703, 'gamma': 0.6860358400076907, 'epsilon': 0.571650312424197}
    # best_params = run_optimization(file_path, seeds[0])
    res = main(file_path, seeds, best_params, cold_start=True)
    res.to_excel("Results6.xlsx", index=False)