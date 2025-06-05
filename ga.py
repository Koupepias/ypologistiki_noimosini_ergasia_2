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

def genetic_algorithm_feature_selection(X, y, trained_weights=None, generations=3, population_size=3, mutation_rate=0.1):
    """
    Genetic Algorithm for feature selection with grouping for categorical features.
    
    Args:
        X (DataFrame): Feature set.
        y (Series): Target variable.
        trained_weights: Pre-trained neural network weights from Part A.
        generations (int): Number of generations to evolve.
        population_size (int): Number of individuals in the population.
        mutation_rate (float): Probability of mutation for each individual.
        
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
        """
        Calculate fitness using masking approach - much faster!
        """
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

    def crossover(parent1, parent2):
        """Crossover that respects feature groups"""
        feature_info = create_feature_groups()
        
        child1 = np.array(parent1.chromosome.copy())
        child2 = np.array(parent2.chromosome.copy())
        
        # Crossover individual features normally
        individual_features = feature_info['individual']
        if len(individual_features) > 1:
            crossover_point = np.random.randint(1, len(individual_features))
            
            # Swap individual features after crossover point
            temp = child1[individual_features[crossover_point:]]
            child1[individual_features[crossover_point:]] = child2[individual_features[crossover_point:]]
            child2[individual_features[crossover_point:]] = temp
        
        # Handle groups: randomly inherit each group from either parent
        for group in feature_info['groups']:
            if np.random.random() < 0.5:
                # Child1 gets parent2's group, Child2 gets parent1's group
                child1[group] = parent2.chromosome[group]
                child2[group] = parent1.chromosome[group]
        
        return Individual(child1), Individual(child2)

    def mutate(individual):
        """Mutation that respects feature groups"""
        feature_info = create_feature_groups()
        mutated = np.array(individual.chromosome.copy())
        
        # Mutate individual features normally
        for idx in feature_info['individual']:
            if np.random.random() < mutation_rate:
                mutated[idx] = 1 - mutated[idx]  # Flip bit
        
        # Mutate groups: either include entire group or exclude it
        for group in feature_info['groups']:
            if np.random.random() < mutation_rate:
                # Flip entire group
                current_state = mutated[group[0]]  # Check first element of group
                new_state = 1 - current_state
                mutated[group] = new_state  # Set all elements in group to same value
        
        return Individual(mutated)

    # Initialize population with group awareness
    population = initialize_population()

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        
        # Evaluate fitness for each individual
        fitness_scores = []
        for individual in population:
            score = fitness(individual, X, y, trained_weights)
            fitness_scores.append(score)
        
        # Handle case where all fitness scores are zero or negative
        min_fitness = min(fitness_scores)
        if min_fitness <= 0:
            # Shift all scores to be positive
            fitness_scores = [score - min_fitness + 0.001 for score in fitness_scores]
        
        # Select individuals for mating pool (roulette wheel selection)
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            print("Warning: All individuals have zero fitness")
            break
            
        probabilities = [score / total_fitness for score in fitness_scores]
        mating_pool = random.choices(population, weights=probabilities, k=population_size)
        
        # Create next generation
        next_generation = []
        for i in range(0, population_size, 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[min(i + 1, population_size - 1)]
            
            # Use group-aware crossover and mutation
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            next_generation.extend([child1, child2])
            # or
            # next_generation.append(mutate(child1))
            # next_generation.append(mutate(child2))
        
        population = next_generation[:population_size]  # Ensure exact population size

    # Find the best individual in the final population
    fitness_scores = [fitness(individual, X, y, trained_weights) for individual in population]
    best_individual = population[fitness_scores.index(max(fitness_scores))]
    best_features = best_individual.get_selected_features()

    print(f"Best feature subset: {best_features}")
    
    # Print which groups were selected - TODO PRINT OUT AFTER SUCCESSFUL DEBUGING
    feature_info = create_feature_groups()
    selected_groups = []
    if any(best_individual.chromosome[feature_info['groups'][0]]):  # Check ethnicity group
        selected_groups.append("Ethnicity")
    if any(best_individual.chromosome[feature_info['groups'][1]]):  # Check education group
        selected_groups.append("EducationLevel")
    
    print(f"Selected categorical groups: {selected_groups}")
    print(f"Total features selected: {len(best_features)} out of {total_features}")
    
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
    
    # Part B: Use GA for feature selection with pre-trained weights
    print("=" * 50)
    print("PART B: Starting genetic algorithm for feature selection...")
    print("=" * 50)
    best_features = genetic_algorithm_feature_selection(X, y, trained_weights=trained_weights)
    
    # Evaluate with selected features
    print("=" * 50)
    print("FINAL EVALUATION: Testing with GA-selected features...")
    print("=" * 50)
    X_selected = X.iloc[:, best_features]
    final_results = train_and_evaluate_model(X_selected, y, trained_weights)
    
    print("Neural network evaluation with GA-selected features completed successfully!")
    print(f"Selected {len(best_features)} features out of {X.shape[1]} total features")