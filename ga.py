import numpy as np
import random
from neural_network import load_and_preprocess_data, train_and_evaluate_model

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


def genetic_algorithm_feature_selection(X, y, trained_weights=None, generations=2, 
                                       population_size=2, mutation_rate=0.1, 
                                       elite_size=2, tournament_size=3):
    """
    Genetic Algorithm for feature selection with:
    - Tournament Selection
    - Uniform Crossover
    - Bit-flip Mutation
    - Elitism
    
    Args:
        X (DataFrame): Feature set.
        y (Series): Target variable.
        trained_weights: Pre-trained neural network weights from Part A.
        generations (int): Number of generations to evolve.
        population_size (int): Number of individuals in the population.
        mutation_rate (float): Probability of mutation for each individual.
        elite_size (int): Number of elite individuals to preserve (1-2 recommended).
        tournament_size (int): Size of tournament for selection (3 recommended).
        
    Returns:
        best_features (list): List of selected features.
    """
    
    feature_info = create_feature_groups()
    total_features = feature_info['total_features']
    
    def initialize_population():
        """Initialize population respecting feature groups for categorical variables"""
        population = []
        
        for _ in range(population_size):
            chromosome = np.zeros(total_features, dtype=int)
            
            # Randomly select individual features (5-25 features which is 15-70% of total features)
            min_features = 5
            max_features = 25
            num_selected = np.random.randint(min_features, max_features)
            num_selected = min(num_selected, len(feature_info['individual']))
            
            selected_features = np.random.choice(
                feature_info['individual'], # number of non-categorial features return from create_feature_groups()
                size=num_selected, # size of the chromosome before the possible selection of group(s)
                replace=False # ensures no duplicates
            )
            chromosome[selected_features] = 1
            
            # Randomly decide which categorical groups to include
            # 30% chance: no groups, 40% chance: one group, 30% chance: both groups
            groups_to_include = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
            
            if groups_to_include > 0:
                selected_groups = np.random.choice(
                    len(feature_info['groups']), 
                    size=groups_to_include, 
                    replace=False
                )
                
                for group_idx in selected_groups:
                    group_features = feature_info['groups'][group_idx]
                    chromosome[group_features] = 1  # Select entire group
            
            population.append(Individual(chromosome))
        
        return population

    def fitness(individual, X_data, y_data, trained_weights):
        """Calculate fitness using masking approach"""
        selected_features = individual.get_selected_features()
        
        if len(selected_features) == 0:
            return 0.0
        
        try:
            # Use masking approach for fast evaluation
            results = train_and_evaluate_model(X_data, y_data, trained_weights, selected_features)
            
            # Extract accuracy from results dictionary
            accuracy = results['val_accuracy'][0] if results['val_accuracy'] else 0.0
        except Exception as e:
            print(f"Error in fitness evaluation: {e}")
            return 0.0
        
        # Feature count penalty
        num_features = len(selected_features)
        feature_penalty = num_features / X_data.shape[1]
        
        # Combined fitness
        alpha = 0.8
        beta = 0.2
        
        # Option 1: Exponential Penalty
        fitness_score = alpha * accuracy - beta * feature_penalty
        
        # Option 2: Pareto-based Approach
        # fitness_score = pareto_rank + crowding_distance
        
        return max(0.0, fitness_score)

    # Initialize population
    population = initialize_population()
    
    # Track best fitness over generations
    best_fitness_history = []

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        
        # Evaluate fitness for each individual
        fitness_scores = []
        for individual in population:
            score = fitness(individual, X, y, trained_weights)
            fitness_scores.append(score)
        
        # Track best fitness
        current_best_fitness = max(fitness_scores)
        best_fitness_history.append(current_best_fitness)
        print(f"  Best fitness: {current_best_fitness:.4f}")
        
        # Handle case where all fitness scores are zero or negative
        min_fitness = min(fitness_scores)
        if min_fitness <= 0:
            fitness_scores = [score - min_fitness + 0.001 for score in fitness_scores]
        
        # ELITISM: Preserve best individuals
        elite_individuals = elitism_selection(population, fitness_scores, elite_size)
        print(f"  Elite individuals preserved: {len(elite_individuals)}")
        
        # Calculate how many new individuals we need to generate
        new_individuals_needed = population_size - len(elite_individuals)
        
        # TOURNAMENT SELECTION: Select parents for reproduction
        mating_pool = tournament_selection(population, fitness_scores, 
                                         new_individuals_needed, tournament_size)
        
        # Create next generation
        next_generation = elite_individuals.copy()  # Start with elite individuals
        
        # Generate new individuals through crossover and mutation
        for i in range(0, new_individuals_needed, 2):
            # Select two parents
            parent1 = mating_pool[i % len(mating_pool)]
            parent2 = mating_pool[(i + 1) % len(mating_pool)]
            
            # UNIFORM CROSSOVER: Create offspring
            child1, child2 = uniform_crossover(parent1, parent2)
            
            # BIT-FLIP MUTATION: Apply mutation
            child1 = bit_flip_mutation(child1, mutation_rate)
            child2 = bit_flip_mutation(child2, mutation_rate)
            
            next_generation.extend([child1, child2])
        
        # Ensure exact population size
        population = next_generation[:population_size]
        
        # Early stopping if no improvement for several generations
        if len(best_fitness_history) > 5:
            recent_improvements = [best_fitness_history[i] - best_fitness_history[i-1] 
                                 for i in range(-5, 0)]
            if all(improvement < 0.001 for improvement in recent_improvements):
                print(f"  Early stopping at generation {generation + 1} (no significant improvement)")
                break

    # Find the best individual in the final population
    fitness_scores = [fitness(individual, X, y, trained_weights) for individual in population]
    best_individual = population[fitness_scores.index(max(fitness_scores))]
    best_features = best_individual.get_selected_features()

    print(f"\nBest feature subset: {best_features}")
    
    # Print which groups were selected
    feature_info = create_feature_groups()
    selected_groups = []
    if any(best_individual.chromosome[feature_info['groups'][0]]):
        selected_groups.append("Ethnicity")
    if any(best_individual.chromosome[feature_info['groups'][1]]):
        selected_groups.append("EducationLevel")
    
    print(f"Selected categorical groups: {selected_groups}")
    print(f"Total features selected: {len(best_features)} out of {total_features}")
    print(f"Final best fitness: {max(fitness_scores):.4f}")
    
    return best_features

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
    trained_weights = results_part_a.get('trained_weights')
    
    if trained_weights is None:
        print("Warning: No trained weights found in results")
        trained_weights = results_part_a.get('model').get_weights() if results_part_a.get('model') else None
    
    print(f"Trained weights extracted successfully: {trained_weights is not None}")
    
    # Part B: Use GA for feature selection with chosen operators
    print("=" * 50)
    print("PART B: Starting genetic algorithm for feature selection...")
    print("=" * 50)
    
    best_features = genetic_algorithm_feature_selection(
        X, y, 
        trained_weights=trained_weights,
        generations=2,        # Increased for better results
        population_size=2,    # Increased for better diversity
        mutation_rate=0.1,     # 10% mutation rate
        elite_size=2,          # Preserve top 2 individuals
        tournament_size=3      # Tournament size of 3
    )
    
    # Evaluate with selected features
    print("=" * 50)
    print("FINAL EVALUATION: Testing with GA-selected features...")
    print("=" * 50)
    X_selected = X.iloc[:, best_features]
    final_results = train_and_evaluate_model(X_selected, y, trained_weights)
    
    print("Neural network evaluation with GA-selected features completed successfully!")
    print(f"Selected {len(best_features)} features out of {X.shape[1]} total features")
    
    # Print feature reduction ratio
    reduction_ratio = (1 - len(best_features) / X.shape[1]) * 100
    print(f"Feature reduction: {reduction_ratio:.1f}%")