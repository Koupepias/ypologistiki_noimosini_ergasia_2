import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from neural_network import load_and_preprocess_data, train_and_evaluate_model, build_neural_network
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        
    def get_selected_features(self):
        return [i for i, bit in enumerate(self.chromosome) if bit == 1]
    
    def __getitem__(self, index):
        return self.chromosome[index]
    
    def __len__(self):
        return len(self.chromosome)

def create_feature_groups():
    """Define which features must be selected together for categorical variables"""
    # Individual features (can be selected independently) - indices 0-28
    individual_features = list(range(29))  # Age through Forgetfulness
    
    # Group features (must be selected as complete groups)
    ethnicity_group = [29, 30, 31, 32]  # Ethnicity_0, 1, 2, 3
    education_group = [33, 34, 35, 36]  # EducationLevel_0, 1, 2, 3
    
    return {
        'individual': individual_features,
        'groups': [ethnicity_group, education_group],
        'total_features': 37
    }

def tournament_selection(population, fitness_scores, num_parents, tournament_size=3):
    """
    Tournament selection - select individuals through tournaments
    
    Args:
        population: List of Individual objects
        fitness_scores: List of fitness values for each individual
        num_parents: Number of parents to select
        tournament_size: Size of each tournament (default=3)
    
    Returns:
        selected_parents: List of selected Individual objects
    """
    selected_parents = []
    
    for _ in range(num_parents):
        # Select random individuals for tournament
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        
        # Find the winner (individual with highest fitness)
        winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        selected_parents.append(population[winner_idx])
    
    return selected_parents

def uniform_crossover(parent1, parent2, crossover_prob=0.5):
    """
    Uniform crossover that respects feature groups
    
    Args:
        parent1, parent2: Parent Individual objects
        crossover_prob: Probability of exchanging each gene (default=0.5)
    
    Returns:
        child1, child2: Two offspring Individual objects
    """
    feature_info = create_feature_groups()
    
    child1 = np.array(parent1.chromosome.copy())
    child2 = np.array(parent2.chromosome.copy())
    
    # Uniform crossover for individual features
    for idx in feature_info['individual']:
        if random.random() < crossover_prob:
            # Exchange genes between children
            child1[idx], child2[idx] = child2[idx], child1[idx]
    
    # Handle groups: randomly inherit each group from either parent
    for group in feature_info['groups']:
        if random.random() < 0.5:
            # Child1 gets parent2's group, Child2 gets parent1's group
            child1[group] = parent2.chromosome[group]
            child2[group] = parent1.chromosome[group]
    
    return Individual(child1), Individual(child2)

def bit_flip_mutation(individual, mutation_rate=0.1):
    """
    Bit-flip mutation that respects feature groups
    
    Args:
        individual: Individual object to mutate
        mutation_rate: Probability of mutation for each gene
    
    Returns:
        mutated_individual: New Individual object with mutations applied
    """
    feature_info = create_feature_groups()
    mutated = np.array(individual.chromosome.copy())
    
    # Mutate individual features with bit-flip
    for idx in feature_info['individual']:
        if random.random() < mutation_rate:
            mutated[idx] = 1 - mutated[idx]  # Flip bit: 0→1, 1→0
    
    # Mutate groups: either include entire group or exclude it
    for group in feature_info['groups']:
        if random.random() < mutation_rate:
            # Flip entire group
            current_state = mutated[group[0]]  # Check first element of group
            new_state = 1 - current_state
            mutated[group] = new_state  # Set all elements in group to same value
    
    return Individual(mutated)

def elitism_selection(population, fitness_scores, elite_size=2):
    """
    Select the best individuals for elitism
    
    Args:
        population: List of Individual objects
        fitness_scores: List of fitness values
        elite_size: Number of elite individuals to preserve
    
    Returns:
        elite_individuals: List of best Individual objects
    """
    if elite_size == 0:
        return []
    
    # Get indices of best individuals
    elite_indices = sorted(range(len(fitness_scores)), 
                          key=lambda i: fitness_scores[i], 
                          reverse=True)[:elite_size]
    
    return [population[i] for i in elite_indices]

def calculate_accuracy(predictions, y_true):
    """
    Calculate accuracy as percentage of correct predictions.
    
    Args:
        predictions: Model predictions (numpy array)
        y_true: True labels (numpy array)
        
    Returns:
        accuracy: Float between 0 and 1
    """
    # Convert to numpy arrays if not already
    predictions = np.array(predictions)
    y_true = np.array(y_true)
    
    # Handle different prediction formats
    if predictions.ndim > 1:
        # If predictions are probabilities/logits, get the class with highest probability
        predicted_classes = np.argmax(predictions, axis=1)
    else:
        # If binary classification with probabilities
        if np.all((predictions >= 0) & (predictions <= 1)):
            predicted_classes = (predictions > 0.5).astype(int)
        else:
            # Already class predictions
            predicted_classes = predictions.astype(int)
    
    # Calculate accuracy
    correct = np.sum(predicted_classes == y_true)
    total = len(y_true)
    accuracy = correct / total
    
    return accuracy

def genetic_algorithm(X, y, trained_weights=None, 
                      max_generations=100, population_size=20, 
                      mutation_rate=0.1, crossover_rate=0.8,
                      tournament_size=3, 
                      no_improvement_generations=10,
                      min_improvement_percent=0.001):
    """
    Enhanced GA with detailed evolution tracking for plotting
    """
    
    # Build the model once and load trained weights
    print("Setting up pre-trained model for fast evaluation...")
    trained_model = build_neural_network(X.shape[1])
    
    if trained_weights is not None:
        trained_model.set_weights(trained_weights)
        print("✅ Loaded pre-trained weights successfully")
    else:
        print("⚠️  No pre-trained weights provided, using random initialization")
    
    # Create train/validation split for consistent evaluation
    from sklearn.model_selection import train_test_split
    train_indices, val_indices = train_test_split(
        range(len(X)), test_size=0.2, random_state=42, stratify=y
    )
    
    feature_info = create_feature_groups()
    total_features = feature_info['total_features']
    
    def initialize_population():
        population = []
        for _ in range(population_size):
            chromosome = np.zeros(total_features, dtype=int)
            
            # Randomly select individual features (5-25 features - 25%-75% of total features)
            min_features = 8
            max_features = 25
            num_selected = np.random.randint(min_features, max_features)
            num_selected = min(num_selected, len(feature_info['individual']))
            
            selected_features = np.random.choice(
                feature_info['individual'],
                size=num_selected,
                replace=False
            )
            chromosome[selected_features] = 1
            
            # Randomly decide which categorical groups to include
            groups_to_include = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
            
            if groups_to_include > 0:
                selected_groups = np.random.choice(
                    len(feature_info['groups']), 
                    size=groups_to_include, 
                    replace=False
                )
                
                for group_idx in selected_groups:
                    group_features = feature_info['groups'][group_idx]
                    chromosome[group_features] = 1
            
            population.append(Individual(chromosome))
        
        return population

    def fitness(individual, X_data, y_data, trained_model, test_indices=None):
        """
        Fast fitness evaluation using pre-trained model with feature masking
        
        Args:
            individual: Individual with chromosome (binary feature selection)
            X_data: Full feature dataset
            y_data: Target labels
            trained_model: Pre-trained keras model
            test_indices: Optional indices for validation set
        
        Returns:
            fitness_score: Combined accuracy and feature penalty score
        """
        selected_features = individual.get_selected_features()
        
        if len(selected_features) == 0:
            return 0.0
        
        try:
            # Use a subset of data for faster evaluation (optional)
            if test_indices is not None:
                X_eval = X_data.iloc[test_indices]
                y_eval = y_data.iloc[test_indices]
            else:
                # Use full dataset or a random sample
                sample_size = min(1000, len(X_data))  # Limit for speed
                sample_indices = np.random.choice(len(X_data), sample_size, replace=False)
                X_eval = X_data.iloc[sample_indices]
                y_eval = y_data.iloc[sample_indices]
            
            # Create feature mask
            feature_mask = np.zeros(X_data.shape[1], dtype=np.float32)
            feature_mask[selected_features] = 1.0
            
            # Apply mask to input data
            X_masked = X_eval.values * feature_mask
            
            # Get predictions from pre-trained model
            predictions = trained_model.predict(X_masked, verbose=0)
            
            # Calculate accuracy
            y_pred_classes = (predictions > 0.5).astype(int).flatten()
            accuracy = np.mean(y_pred_classes == y_eval.values)
            
            # Feature count penalty
            num_features = len(selected_features)
            feature_penalty = num_features / X_data.shape[1]
            
            # Combined fitness
            alpha = 0.8
            beta = 0.2
            fitness_score = alpha * accuracy - beta * feature_penalty
            
            return max(0.0, fitness_score)
            
        except Exception as e:
            print(f"Error in fast fitness evaluation: {e}")
            return 0.0

    # Initialize population and tracking variables
    population = initialize_population()
    
    # ENHANCED TRACKING FOR PLOTTING
    evolution_data = {
        'generation': [],
        'best_fitness': [],
        'average_fitness': [],
        'worst_fitness': [],
        'std_fitness': [],
        'diversity': [],
        'feature_count_best': [],
        'feature_count_avg': [],
        'improvement_rate': []
    }
    
    generation_count = 0
    termination_reason = ""
    
    # Main evolution loop
    for generation in range(max_generations):
        print(f"=== GENERATION {generation + 1}/{max_generations} ===")
        generation_count = generation + 1
        
        # Fast fitness evaluation for each individual
        fitness_scores = []
        for i, individual in enumerate(population):
            print(f"  Evaluating individual {i+1}/{len(population)}")
            # Pass the trained model and validation indices
            score = fitness(individual, X, y, trained_model, val_indices)
            fitness_scores.append(score)
        
        # Calculate statistics
        current_best_fitness = max(fitness_scores)
        current_avg_fitness = np.mean(fitness_scores)
        current_worst_fitness = min(fitness_scores)
        current_std_fitness = np.std(fitness_scores)
        
        # Calculate diversity
        diversity = calculate_population_diversity(population)
        
        # Calculate feature counts
        feature_counts = [len(ind.get_selected_features()) for ind in population]
        best_individual_idx = fitness_scores.index(current_best_fitness)
        best_feature_count = feature_counts[best_individual_idx]
        avg_feature_count = np.mean(feature_counts)
        
        # Calculate improvement rate
        if generation > 0:
            prev_best = evolution_data['best_fitness'][-1]
            improvement_rate = ((current_best_fitness - prev_best) / max(prev_best, 0.001)) * 100
        else:
            improvement_rate = 0.0
        
        # Store evolution data
        evolution_data['generation'].append(generation_count)
        evolution_data['best_fitness'].append(current_best_fitness)
        evolution_data['average_fitness'].append(current_avg_fitness)
        evolution_data['worst_fitness'].append(current_worst_fitness)
        evolution_data['std_fitness'].append(current_std_fitness)
        evolution_data['diversity'].append(diversity)
        evolution_data['feature_count_best'].append(best_feature_count)
        evolution_data['feature_count_avg'].append(avg_feature_count)
        evolution_data['improvement_rate'].append(improvement_rate)
        
        print(f"Gen {generation_count:3d}: Best={current_best_fitness:.4f}, "
              f"Avg={current_avg_fitness:.4f}, Diversity={diversity:.3f}")
        
        # TERMINATION CRITERIA CHECKS
        if len(evolution_data['best_fitness']) >= no_improvement_generations:
            recent_best = evolution_data['best_fitness'][-no_improvement_generations:]
            max_recent = max(recent_best)
            min_recent = min(recent_best)
            if abs(max_recent - min_recent) < 1e-6:  # Better comparison
                termination_reason = f"No improvement for {no_improvement_generations} generations"
                print(f"🛑 TERMINATING: {termination_reason}")
                break
        
        # Check for minimal improvement
        if len(evolution_data['best_fitness']) >= 5:
            improvement_window = 5
            old_fitness = evolution_data['best_fitness'][-improvement_window]
            improvement_percent = ((current_best_fitness - old_fitness) / max(old_fitness, 0.001)) * 100
            
            if improvement_percent < min_improvement_percent:
                termination_reason = f"Improvement below {min_improvement_percent}% threshold"
                print(f"🛑 TERMINATING: {termination_reason}")
                break
        
        # ✅ ADD SAFETY CHECK TO PREVENT INFINITE LOOPS
        if generation_count >= max_generations:
            termination_reason = f"Maximum generations ({max_generations}) reached"
            print(f"🛑 TERMINATING: {termination_reason}")
            break
        
        # GENETIC OPERATIONS (only if not terminating)
        elite_individuals = elitism_selection(population, fitness_scores, 2)
        new_individuals_needed = population_size - len(elite_individuals)
        mating_pool = tournament_selection(population, fitness_scores, 
                                         new_individuals_needed, tournament_size)
        
        next_generation = elite_individuals.copy()
        
        for i in range(0, new_individuals_needed, 2):
            parent1 = mating_pool[i % len(mating_pool)]
            parent2 = mating_pool[(i + 1) % len(mating_pool)]
            
            if random.random() < crossover_rate:
                child1, child2 = uniform_crossover(parent1, parent2)
            else:
                child1 = Individual(np.array(parent1.chromosome.copy()))
                child2 = Individual(np.array(parent2.chromosome.copy()))
            
            child1 = bit_flip_mutation(child1, mutation_rate)
            child2 = bit_flip_mutation(child2, mutation_rate)
            
            next_generation.extend([child1, child2])
        
        population = next_generation[:population_size]
    
    # If loop completed without break
    if not termination_reason:
        termination_reason = f"Maximum generations ({max_generations}) reached"
    
    # Final evaluation
    final_fitness_scores = [fitness(individual, X, y, trained_model, val_indices) for individual in population]  # Fixed: use trained_model instead of trained_weights
    best_individual = population[final_fitness_scores.index(max(final_fitness_scores))]
    best_features = best_individual.get_selected_features()
    final_best_fitness = max(final_fitness_scores)
    
    # Return comprehensive results including evolution data
    results = {
        'best_features': best_features,
        'final_best_fitness': final_best_fitness,
        'generations_run': generation_count,
        'termination_reason': termination_reason,
        'evolution_data': evolution_data,  # NEW: Detailed evolution tracking
        'final_feature_count': len(best_features)
    }
    
    return results

def calculate_population_diversity(population):
    """Calculate average Hamming distance between individuals (normalized)"""
    if len(population) < 2:
        return 0.0
    
    total_distance = 0
    count = 0
    
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            # Hamming distance
            distance = sum(a != b for a, b in zip(population[i].chromosome, population[j].chromosome))
            total_distance += distance
            count += 1
    
    if count == 0:
        return 0.0
    
    avg_distance = total_distance / count
    max_possible_distance = len(population[0].chromosome)
    
    return avg_distance / max_possible_distance

def run_multiple_ga_experiments(X, y, trained_weights, num_runs=10, **ga_params):
    """
    Run GA multiple times and collect statistics INCLUDING evolution data
    
    Args:
        X, y: Dataset
        trained_weights: Pre-trained model weights
        num_runs: Number of independent runs (default=10)
        **ga_params: Parameters for GA
    
    Returns:
        experiment_results: Dictionary with aggregated statistics AND evolution data
    """
    
    print(f"Running {num_runs} independent GA experiments...")
    print(f"Parameters: {ga_params}")
    print("=" * 60)
    
    all_results = []
    all_evolution_data = []  # NEW: Collect evolution data from all runs
    
    for run in range(num_runs):
        print(f"\n--- RUN {run + 1}/{num_runs} ---")
        
        # Set different random seed for each run
        random.seed(42 + run)
        np.random.seed(42 + run)
        
        # Run GA
        result = genetic_algorithm(
            X, y, trained_weights, **ga_params
        )
        
        result['run_number'] = run + 1
        all_results.append(result)
        
        # NEW: Collect evolution data from this run
        if 'evolution_data' in result:
            all_evolution_data.append(result['evolution_data'])
        
        print(f"Run {run + 1} completed:")
        print(f"  Best fitness: {result['final_best_fitness']:.4f}")
        print(f"  Features: {result['final_feature_count']}")
        print(f"  Generations: {result['generations_run']}")
        print(f"  Termination: {result['termination_reason']}")
    
    # NEW: Process evolution data for averaging across runs
    processed_evolution_data = process_evolution_data_for_plotting(all_evolution_data)
    
    # Aggregate statistics (same as before)
    best_fitnesses = [r['final_best_fitness'] for r in all_results]
    feature_counts = [r['final_feature_count'] for r in all_results]
    generations = [r['generations_run'] for r in all_results]
    
    experiment_results = {
        'individual_runs': all_results,
        'evolution_data': processed_evolution_data,  # NEW: Add processed evolution data
        'statistics': {
            'num_runs': num_runs,
            'best_fitness': {
                'mean': np.mean(best_fitnesses),
                'std': np.std(best_fitnesses),
                'min': np.min(best_fitnesses),
                'max': np.max(best_fitnesses),
                'median': np.median(best_fitnesses)
            },
            'feature_count': {
                'mean': np.mean(feature_counts),
                'std': np.std(feature_counts),
                'min': int(np.min(feature_counts)),
                'max': int(np.max(feature_counts)),
                'median': np.median(feature_counts)
            },
            'generations': {
                'mean': np.mean(generations),
                'std': np.std(generations),
                'min': int(np.min(generations)),
                'max': int(np.max(generations)),
                'median': np.median(generations)
            }
        },
        'termination_reasons': {reason: sum(1 for r in all_results if r['termination_reason'] == reason) 
                               for reason in set(r['termination_reason'] for r in all_results)},
        'parameters_used': ga_params
    }
    
    return experiment_results

def process_evolution_data_for_plotting(all_evolution_data):
    """
    Process evolution data from multiple runs for plotting
    """
    
    if not all_evolution_data:
        print("⚠️  No evolution data to process")
        return {}
    
    # Find the maximum number of generations across all runs
    max_generations = max(len(data['generation']) for data in all_evolution_data)
    
    # Initialize arrays for averaging
    metrics = ['best_fitness', 'average_fitness', 'worst_fitness', 'diversity', 
               'feature_count_best', 'feature_count_avg']
    
    processed_data = {metric: {'mean': [], 'std': [], 'min': [], 'max': [], 'all_runs': []} 
                     for metric in metrics}
    processed_data['generation'] = list(range(1, max_generations + 1))
    
    print(f"Processing evolution data from {len(all_evolution_data)} runs, max generations: {max_generations}")
    
    # For each generation, collect data from all runs
    for gen in range(max_generations):
        gen_data = {metric: [] for metric in metrics}
        
        # Collect data from all runs for this generation
        for run_data in all_evolution_data:
            if gen < len(run_data['generation']):  # Run didn't terminate yet
                for metric in metrics:
                    if metric in run_data:
                        gen_data[metric].append(run_data[metric][gen])
            else:  # Run already terminated, use last value
                for metric in metrics:
                    if metric in run_data and run_data[metric]:
                        gen_data[metric].append(run_data[metric][-1])
        
        # Calculate statistics for this generation
        for metric in metrics:
            if gen_data[metric]:  # If we have data for this generation
                processed_data[metric]['mean'].append(np.mean(gen_data[metric]))
                processed_data[metric]['std'].append(np.std(gen_data[metric]))
                processed_data[metric]['min'].append(np.min(gen_data[metric]))
                processed_data[metric]['max'].append(np.max(gen_data[metric]))
                processed_data[metric]['all_runs'].append(gen_data[metric])
            else:
                # No data available, use previous value or 0
                prev_val = processed_data[metric]['mean'][-1] if processed_data[metric]['mean'] else 0
                processed_data[metric]['mean'].append(prev_val)
                processed_data[metric]['std'].append(0)
                processed_data[metric]['min'].append(prev_val)
                processed_data[metric]['max'].append(prev_val)
                processed_data[metric]['all_runs'].append([prev_val])
    
    print(f"✅ Evolution data processed successfully")
    return processed_data

def analyze_and_report_results(experiment_results):
    """Generate comprehensive analysis report"""
    
    stats = experiment_results['statistics']
    
    print("\n" + "="*80)
    print("COMPREHENSIVE GA ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nEXPERIMENT PARAMETERS:")
    for param, value in experiment_results['parameters_used'].items():
        print(f"  {param}: {value}")
    
    print(f"\nFITNESS STATISTICS (over {stats['num_runs']} runs):")
    print(f"  Mean ± Std:     {stats['best_fitness']['mean']:.4f} ± {stats['best_fitness']['std']:.4f}")
    print(f"  Best (Max):     {stats['best_fitness']['max']:.4f}")
    print(f"  Worst (Min):    {stats['best_fitness']['min']:.4f}")
    print(f"  Median:         {stats['best_fitness']['median']:.4f}")
    
    print(f"\nFEATURE COUNT STATISTICS:")
    print(f"  Mean ± Std:     {stats['feature_count']['mean']:.1f} ± {stats['feature_count']['std']:.1f}")
    print(f"  Min features:   {stats['feature_count']['min']}")
    print(f"  Max features:   {stats['feature_count']['max']}")
    print(f"  Median:         {stats['feature_count']['median']:.1f}")
    
    print(f"\nCONVERGENCE STATISTICS:")
    print(f"  Mean generations: {stats['generations']['mean']:.1f} ± {stats['generations']['std']:.1f}")
    print(f"  Fastest:          {stats['generations']['min']} generations")
    print(f"  Slowest:          {stats['generations']['max']} generations")
    print(f"  Median:           {stats['generations']['median']:.1f} generations")
    
    print(f"\nTERMINATION REASONS:")
    for reason, count in experiment_results['termination_reasons'].items():
        percentage = (count / stats['num_runs']) * 100
        print(f"  {reason}: {count}/{stats['num_runs']} runs ({percentage:.1f}%)")
    
    # Consistency analysis
    cv_fitness = stats['best_fitness']['std'] / stats['best_fitness']['mean']
    cv_features = stats['feature_count']['std'] / stats['feature_count']['mean']
    
    print(f"\nCONSISTENCY ANALYSIS:")
    print(f"  Fitness CV:     {cv_fitness:.3f} {'(Low variance)' if cv_fitness < 0.1 else '(High variance)'}")
    print(f"  Features CV:    {cv_features:.3f} {'(Consistent)' if cv_features < 0.2 else '(Variable)'}")
    
    return stats

def save_results_to_files(experiment_results, base_filename="ga_experiment", AA=1):
    """Save results to JSON and CSV files with evolution data support"""
    
    # Create directory if it doesn't exist
    folder = f'parameters_evaluation_{AA}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    print(f"Created results directory: {folder}")
    
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    try:
        # Save detailed results to JSON
        json_filename = f"{folder}/{base_filename}.json"
        
        # Convert all numpy types to native Python types
        json_safe_results = convert_numpy_types(experiment_results)
        
        with open(json_filename, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        print(f"✅ Detailed results saved to: {json_filename}")
        
        # Save summary statistics to CSV
        csv_filename = f"{folder}/{base_filename}_summary.csv"
        summary_data = []
        
        for i, run in enumerate(experiment_results['individual_runs'], 1):
            summary_data.append({
                'run': i,
                'best_fitness': float(run['final_best_fitness']),
                'feature_count': int(run['final_feature_count']),
                'generations': int(run['generations_run']),
                'termination_reason': str(run['termination_reason'])
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(csv_filename, index=False)
        print(f"✅ Summary CSV saved to: {csv_filename}")
        
        # NEW: Save evolution data CSV (if available)
        if 'evolution_data' in experiment_results and experiment_results['evolution_data']:
            evolution_csv_filename = f"{folder}/{base_filename}_evolution.csv"
            
            evolution_data = experiment_results['evolution_data']
            
            # Create DataFrame with proper type conversion
            evolution_df_data = {}
            
            # Add generation column
            if 'generation' in evolution_data:
                evolution_df_data['generation'] = [int(x) for x in evolution_data['generation']]
            
            # Add all metrics with mean and std
            for key in ['best_fitness', 'average_fitness', 'worst_fitness', 'diversity', 
                       'feature_count_best', 'feature_count_avg']:
                if key in evolution_data and 'mean' in evolution_data[key]:
                    evolution_df_data[f'{key}_mean'] = [float(x) for x in evolution_data[key]['mean']]
                    evolution_df_data[f'{key}_std'] = [float(x) for x in evolution_data[key]['std']]
                    evolution_df_data[f'{key}_min'] = [float(x) for x in evolution_data[key]['min']]
                    evolution_df_data[f'{key}_max'] = [float(x) for x in evolution_data[key]['max']]
            
            if evolution_df_data:  # Only create if we have data
                evolution_df = pd.DataFrame(evolution_df_data)
                evolution_df.to_csv(evolution_csv_filename, index=False)
                print(f"✅ Evolution data CSV saved to: {evolution_csv_filename}")
        
        print(f"\n✅ All results successfully saved to folder: {folder}")
        
    except Exception as e:
        print(f"❌ Error saving results: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Try to save at least the summary data
        try:
            fallback_filename = f"{folder}/{base_filename}_fallback_summary.csv"
            summary_data = []
            for i, run in enumerate(experiment_results['individual_runs'], 1):
                summary_data.append({
                    'run': i,
                    'best_fitness': float(run['final_best_fitness']),
                    'feature_count': int(run['final_feature_count']),
                    'generations': int(run['generations_run']),
                    'termination_reason': str(run['termination_reason'])
                })
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_csv(fallback_filename, index=False)
            print(f"✅ Fallback summary saved to: {fallback_filename}")
            
        except Exception as fallback_error:
            print(f"❌ Fallback save also failed: {str(fallback_error)}")
    
    return folder

def plot_evolution_curves(experiment_results, save_folder, figsize=(10, 6)):
    """
    Create comprehensive evolution curve plots with safe data handling
    Each plot will be saved as a separate figure
    """
    
    # Check if evolution data exists
    if 'evolution_data' not in experiment_results or not experiment_results['evolution_data']:
        print("⚠️  No evolution data found in experiment results.")
        print("   Creating simplified summary plots instead...")
        return plot_summary_results(experiment_results, f"{save_folder}/summary_results.png", figsize)
    
    evolution_data = experiment_results['evolution_data']
    
    # Check if we have the required data
    if 'generation' not in evolution_data or 'best_fitness' not in evolution_data:
        print("⚠️  Incomplete evolution data found.")
        return plot_summary_results(experiment_results, f"{save_folder}/summary_results.png", figsize)
    
    # Create folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Created plots directory: {save_folder}")
    
    generations = evolution_data['generation']
    
    # Store all figures to return them
    figures = {}
    
    # Plot 1: Best Fitness Evolution (Main Plot)
    fig1 = plt.figure(figsize=figsize)
    ax1 = fig1.add_subplot(111)
    
    # Plot individual runs (light lines) - if available
    for i, run_result in enumerate(experiment_results['individual_runs']):
        if 'evolution_data' in run_result:
            run_evolution = run_result['evolution_data']
            ax1.plot(run_evolution['generation'], run_evolution['best_fitness'], 
                    alpha=0.2, color='lightblue', linewidth=0.5)
    
    # Plot average with confidence interval
    mean_fitness = evolution_data['best_fitness']['mean']
    std_fitness = evolution_data['best_fitness']['std']
    
    ax1.plot(generations, mean_fitness, 'b-', linewidth=2, label='Average Best Fitness')
    ax1.fill_between(generations, 
                     np.array(mean_fitness) - np.array(std_fitness),
                     np.array(mean_fitness) + np.array(std_fitness),
                     alpha=0.3, color='blue', label='±1 Std Dev')
    
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Best Fitness Evolution\n(Average across {} runs)'.format(
        experiment_results['statistics']['num_runs']))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/best_fitness_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()
    figures['best_fitness'] = fig1
    
    # Plot 2: Population Fitness Statistics
    fig2 = plt.figure(figsize=figsize)
    ax2 = fig2.add_subplot(111)
    
    ax2.plot(generations, evolution_data['best_fitness']['mean'], 'g-', 
             linewidth=2, label='Best')
    
    if 'average_fitness' in evolution_data:
        ax2.plot(generations, evolution_data['average_fitness']['mean'], 'b-', 
                 linewidth=2, label='Average')
    
    if 'worst_fitness' in evolution_data:
        ax2.plot(generations, evolution_data['worst_fitness']['mean'], 'r-', 
                 linewidth=2, label='Worst')
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness')
    ax2.set_title('Population Fitness Statistics\n(Average across {} runs)'.format(
        experiment_results['statistics']['num_runs']))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/population_fitness_stats.png", dpi=300, bbox_inches='tight')
    plt.close()
    figures['population_stats'] = fig2
    
    # Plot 3: Population Diversity (if available)
    fig3 = plt.figure(figsize=figsize)
    ax3 = fig3.add_subplot(111)
    
    if 'diversity' in evolution_data and evolution_data['diversity']['mean']:
        diversity_mean = evolution_data['diversity']['mean']
        diversity_std = evolution_data['diversity']['std']
        
        ax3.plot(generations, diversity_mean, 'purple', linewidth=2, label='Average Diversity')
        ax3.fill_between(generations, 
                         np.array(diversity_mean) - np.array(diversity_std),
                         np.array(diversity_mean) + np.array(diversity_std),
                         alpha=0.3, color='purple')
        
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Diversity (Normalized Hamming Distance)')
        ax3.set_title('Population Diversity\n(Average across {} runs)'.format(
            experiment_results['statistics']['num_runs']))
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Diversity data\nnot available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Population Diversity (Not Available)')
    
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/population_diversity.png", dpi=300, bbox_inches='tight')
    plt.close()
    figures['diversity'] = fig3
    
    # Plot 4: Feature Count Evolution (if available)
    fig4 = plt.figure(figsize=figsize)
    ax4 = fig4.add_subplot(111)
    
    if 'feature_count_best' in evolution_data and evolution_data['feature_count_best']['mean']:
        ax4.plot(generations, evolution_data['feature_count_best']['mean'], 'orange', 
                 linewidth=2, label='Best Individual')
        
        if 'feature_count_avg' in evolution_data:
            ax4.plot(generations, evolution_data['feature_count_avg']['mean'], 'red', 
                     linewidth=2, label='Population Average')
        
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Number of Features')
        ax4.set_title('Feature Count Evolution\n(Average across {} runs)'.format(
            experiment_results['statistics']['num_runs']))
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Feature count data\nnot available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Feature Count Evolution (Not Available)')
    
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/feature_count_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()
    figures['feature_count'] = fig4
    
    # Plot 5: Convergence Analysis
    fig5 = plt.figure(figsize=figsize)
    ax5 = fig5.add_subplot(111)
    
    # Calculate improvement rate
    improvement_rates = []
    mean_fitness = evolution_data['best_fitness']['mean']
    for i in range(1, len(mean_fitness)):
        prev_fitness = mean_fitness[i-1]
        curr_fitness = mean_fitness[i]
        improvement = ((curr_fitness - prev_fitness) / max(prev_fitness, 0.001)) * 100
        improvement_rates.append(improvement)
    
    if improvement_rates:
        ax5.plot(generations[1:], improvement_rates, 'darkgreen', linewidth=2)
        ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, 
                    label='1% Improvement Threshold')
        
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('Improvement Rate (%)')
        ax5.set_title('Fitness Improvement Rate\n(Average across {} runs)'.format(
            experiment_results['statistics']['num_runs']))
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'Improvement rate\ncannot be calculated', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Fitness Improvement Rate (Not Available)')
    
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/improvement_rate.png", dpi=300, bbox_inches='tight')
    plt.close()
    figures['improvement_rate'] = fig5
    
    # Plot 6: Termination Analysis
    fig6 = plt.figure(figsize=figsize)
    ax6 = fig6.add_subplot(111)
    
    if 'termination_reasons' in experiment_results:
        termination_reasons = experiment_results['termination_reasons']
        reasons = list(termination_reasons.keys())
        counts = list(termination_reasons.values())
        
        # Shorten reason names for display
        short_reasons = []
        for reason in reasons:
            if 'No improvement' in reason:
                short_reasons.append('No Improvement')
            elif 'below' in reason:
                short_reasons.append('Low Improvement')
            elif 'Maximum' in reason:
                short_reasons.append('Max Generations')
            else:
                short_reasons.append(reason[:15])
        
        bars = ax6.bar(short_reasons, counts, color=['skyblue', 'lightcoral', 'lightgreen'][:len(counts)])
        ax6.set_ylabel('Number of Runs')
        ax6.set_title('Termination Reasons\n({} total runs)'.format(
            experiment_results['statistics']['num_runs']))
        ax6.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
    else:
        ax6.text(0.5, 0.5, 'Termination data\nnot available', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Termination Reasons (Not Available)')
    
    plt.tight_layout()
    plt.savefig(f"{save_folder}/termination_reasons.png", dpi=300, bbox_inches='tight')
    plt.close()
    figures['termination'] = fig6
    
    # Close all figures
    plt.close('all')
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EVOLUTION CURVE ANALYSIS")
    print("="*60)
    
    final_best_mean = evolution_data['best_fitness']['mean'][-1]
    final_best_std = evolution_data['best_fitness']['std'][-1]
    
    print(f"Final Best Fitness: {final_best_mean:.4f} ± {final_best_std:.4f}")
    
    if improvement_rates:
        # Find generation with steepest improvement
        max_improvement_gen = np.argmax(improvement_rates) + 2  # +2 because we start from generation 2
        max_improvement_rate = max(improvement_rates)
        
        print(f"Steepest improvement at generation {max_improvement_gen}: {max_improvement_rate:.2f}%")
        
        # Calculate convergence generation (when improvement rate drops below 1%)
        convergence_gen = None
        for i, rate in enumerate(improvement_rates):
            if rate < 1.0:
                convergence_gen = i + 2
                break
        
        if convergence_gen:
            print(f"Convergence generation (improvement < 1%): {convergence_gen}")
        else:
            print("No clear convergence point found")
    
    print(f"✅ All plots saved to directory: {save_folder}")
    
    return figures

def plot_summary_results(experiment_results, save_path="summary_results.png", figsize=(12, 8)):
    """
    Create summary plots when evolution data is not available
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Genetic Algorithm Results Summary\n({} runs)'.format(
        experiment_results['statistics']['num_runs']), fontsize=16, fontweight='bold')
    
    # Plot 1: Final Fitness Distribution
    ax1 = axes[0, 0]
    
    final_fitnesses = [r['final_best_fitness'] for r in experiment_results['individual_runs']]
    
    ax1.hist(final_fitnesses, bins=min(10, len(final_fitnesses)), 
             alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(final_fitnesses), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(final_fitnesses):.4f}')
    ax1.axvline(np.median(final_fitnesses), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(final_fitnesses):.4f}')
    
    ax1.set_xlabel('Final Best Fitness')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Final Fitness Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Feature Count Distribution
    ax2 = axes[0, 1]
    
    feature_counts = [r['final_feature_count'] for r in experiment_results['individual_runs']]
    
    ax2.hist(feature_counts, bins=min(10, len(feature_counts)), 
             alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axvline(np.mean(feature_counts), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(feature_counts):.1f}')
    
    ax2.set_xlabel('Number of Features Selected')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Feature Count Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Generations Used Distribution  
    ax3 = axes[1, 0]
    
    generations_used = [r['generations_run'] for r in experiment_results['individual_runs']]
    
    ax3.hist(generations_used, bins=min(10, len(generations_used)), 
             alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.axvline(np.mean(generations_used), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(generations_used):.1f}')
    
    ax3.set_xlabel('Generations Until Termination')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Convergence Speed Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Feature Count vs Fitness Scatter
    ax4 = axes[1, 1]
    
    scatter = ax4.scatter(feature_counts, final_fitnesses, 
                         c=range(len(final_fitnesses)), cmap='viridis', 
                         alpha=0.7, s=100, edgecolors='black')
    
    # Add trend line
    if len(feature_counts) > 1:
        z = np.polyfit(feature_counts, final_fitnesses, 1)
        p = np.poly1d(z)
        ax4.plot(sorted(feature_counts), p(sorted(feature_counts)), 
                 "r--", alpha=0.8, linewidth=2, label=f'Trend: slope={z[0]:.4f}')
        ax4.legend()
    
    ax4.set_xlabel('Number of Features Selected')
    ax4.set_ylabel('Final Best Fitness')
    ax4.set_title('Feature Count vs Performance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.colorbar(scatter, ax=ax4, label='Run Number')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print("SUMMARY RESULTS ANALYSIS")
    print("="*60)
    print(f"Final Best Fitness: {np.mean(final_fitnesses):.4f} ± {np.std(final_fitnesses):.4f}")
    print(f"Average Feature Count: {np.mean(feature_counts):.1f} ± {np.std(feature_counts):.1f}")
    print(f"Average Generations: {np.mean(generations_used):.1f} ± {np.std(generations_used):.1f}")
    
    return fig

if __name__ == "__main__":
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    # Part A: Train neural network on full feature set and get weights
    print("=" * 50)
    print("PART A: Training neural network on full feature set...")
    print("=" * 50)
    results_part_a = train_and_evaluate_model(X, y)
    
    # Extract trained weights more robustly
    trained_weights = None
    if 'trained_weights' in results_part_a and results_part_a['trained_weights'] is not None:
        trained_weights = results_part_a['trained_weights']
    elif 'model' in results_part_a and results_part_a['model'] is not None:
        trained_weights = results_part_a['model'].get_weights()
    else:
        print("⚠️  Warning: No trained weights found, will train a new model")
        # Train a simple model to get weights
    
    print(f"Trained weights extracted successfully: {trained_weights is not None}")
    
    # Part B2: Multiple runs with termination criteria (including crossover rate)
    print("=" * 50)
    print("PART B2: GA with Termination Criteria - Multiple Runs (with Crossover Rate)")
    print("=" * 50)
    
    # Define GA parameters including crossover rate
    ga_parameters = {
        'max_generations': 100,
        'population_size': 20,
        'mutation_rate': 0.00,
        'crossover_rate': 0.6,      
        # 'elite_size': 2,
        'tournament_size': 3,
        'no_improvement_generations': 10,
        'min_improvement_percent': 0.001
    }
    
    # Run 10 independent experiments
    experiment_results = run_multiple_ga_experiments(
        X, y, trained_weights, 
        num_runs=10,
        **ga_parameters
    )
    
    # Analyze results
    stats = analyze_and_report_results(experiment_results)
    
    # parameters selection row
    AA=1
    # Save results
    folder = save_results_to_files(experiment_results, "ga_with_crossover_rate", AA)
    
    # Generate all plots
    print("\n" + "="*60)
    print("GENERATING EVOLUTION CURVE PLOTS")
    print("="*60)
    
    # Main evolution curves plot
    fig1 = plot_evolution_curves(experiment_results, folder)
    
    # Part C: Train model with best selected features
    print("\n" + "="*60)
    print("PART C: TRAINING MODEL WITH SELECTED FEATURES")
    print("="*60)
    
    # Get the best individual across all runs
    best_run = max(experiment_results['individual_runs'], 
                  key=lambda x: x['final_best_fitness'])
    
    best_features = best_run['best_features']
    print(f"Best features selected: {len(best_features)} features")
    print(f"Feature indices: {best_features}")
    
    # Get feature names for better interpretability
    feature_names = list(X.columns)
    best_feature_names = [feature_names[i] for i in best_features]
    print("\nSelected feature names:")
    for i, feature in enumerate(best_feature_names):
        print(f"  {i+1}. {feature}")
    
    # Create dataset with only selected features
    X_selected = X.iloc[:, best_features]
    print(f"\nSelected features dataset shape: {X_selected.shape}")
    
    # Train and evaluate model with selected features
    print("\nTraining neural network with selected features...")
    results_selected = train_and_evaluate_model(X_selected, y)
    
    # Compare with full-feature model
    print("\n" + "="*60)
    print("COMPARISON: FULL FEATURES vs. SELECTED FEATURES")
    print("="*60)
    print(f"Full features ({X.shape[1]}): Accuracy = {results_part_a['accuracy']:.4f}")
    print(f"Selected features ({len(best_features)}): Accuracy = {results_selected['accuracy']:.4f}")
    print(f"Feature reduction: {(1 - len(best_features)/X.shape[1])*100:.1f}%")
    
    # Save comparison results
    comparison = {
        "full_features": {
            "count": X.shape[1],
            "accuracy": float(results_part_a['accuracy']),
        },
        "selected_features": {
            "count": len(best_features),
            "accuracy": float(results_selected['accuracy']),
            "feature_indices": best_features,
            "feature_names": best_feature_names
        },
        "improvement": {
            "accuracy_change": float(results_selected['accuracy'] - results_part_a['accuracy']),
            "feature_reduction_percent": float((1 - len(best_features)/X.shape[1])*100)
        }
    }
    
    # Save comparison to JSON
    with open(f"{folder}/feature_selection_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n✅ Comparison results saved to: {folder}/feature_selection_comparison.json")
    
    # Create comparison bar chart
    plt.figure(figsize=(10, 6))
    
    models = ['Full Features', 'Selected Features']
    accuracies = [results_part_a['accuracy'], results_selected['accuracy']]
    feature_counts = [X.shape[1], len(best_features)]
    
    # Plot accuracy bars
    ax1 = plt.subplot(111)
    bars = ax1.bar(models, accuracies, alpha=0.7, color=['skyblue', 'lightgreen'])
    ax1.set_ylim([0.75, 1.0])  # Adjust as needed to make differences visible
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance Comparison')
    
    # Add feature counts as text
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'Acc: {accuracies[i]:.4f}', ha='center', va='bottom')
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                f'Features: {feature_counts[i]}', ha='center', va='center', 
                color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{folder}/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comparison chart saved to: {folder}/model_comparison.png")
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
