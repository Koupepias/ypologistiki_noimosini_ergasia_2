import numpy as np
import random
from sklearn.model_selection import train_test_split
from neural_network import train_and_evaluate_model

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        
    def get_selected_features(self):
        return [i for i, bit in enumerate(self.chromosome) if bit == 1]
    
    def __getitem__(self, index):
        return self.chromosome[index]
    
    def __len__(self):
        return len(self.chromosome)

def genetic_algorithm_feature_selection(X, y, generations=20, population_size=10, mutation_rate=0.1):
    """
    Genetic Algorithm for feature selection.
    
    Args:
        X (DataFrame): Feature set.
        y (Series): Target variable.
        generations (int): Number of generations to evolve.
        population_size (int): Number of individuals in the population.
        mutation_rate (float): Probability of mutation for each individual.
        
    Returns:
        best_features (list): List of selected features.
    """
    def initialize_population(num_features=32, min_features=5, max_features=25):
        """Initialize the population with random feature subsets."""
        population = []
        for i in range(population_size):
            # Determine number of features (8-20 range)
            min_features = 8
            max_features = 20
            num_selected = np.random.randint(min_features, max_features + 1)
            
            # Create binary chromosome
            chromosome = np.zeros(num_features, dtype=int)
            selected_features = np.random.choice(num_features, num_selected, replace=False)
            chromosome[selected_features] = 1
            
            population.append(Individual(chromosome))
        return population

    def fitness(individual, X_test, y_test, trained_weights):
        """
        Calculate fitness combining accuracy and feature count penalty
        
        Args:
            individual: GA individual with binary chromosome
            X_test: test data
            y_test: test labels  
            trained_weights: pre-trained NN weights from part A
        """
        # Extract selected features
        selected_features = individual.get_selected_features()
        X_selected = X_test[:, selected_features]
        
        # Evaluate NN with selected features
        predictions = train_and_evaluate_model(X_selected, y, trained_weights, selected_features)
        accuracy = calculate_accuracy(predictions, y_test)
        
        # Feature count penalty
        num_features = len(selected_features)
        feature_penalty = num_features / 34.0  # Normalized to [0,1]
        
        # Combined fitness (higher is better)
        alpha = 0.8  # Weight for accuracy (primary objective)
        beta = 0.2   # Weight for feature reduction (secondary objective)
        
        # Option 1: Exponential Penalty
        fitness = alpha * accuracy - beta * feature_penalty
        # Option 2: Pareto-based Approach
        # fitness = pareto_rank + crowding_distance
        
        return fitness

    def crossover(parent1, parent2):
        """Perform crossover between two parents."""
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(individual):
        """Perform mutation on an individual."""
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = 1 - individual[i]  # Flip the bit
        return individual

    # Initialize population
    population = initialize_population()

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        
        # Evaluate fitness for each individual
        fitness_scores = [fitness(individual) for individual in population]
        
        # Select individuals for mating pool (roulette wheel selection)
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        mating_pool = random.choices(population, probabilities, k=population_size)
        
        # Create next generation
        next_generation = []
        for i in range(0, population_size, 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[i + 1]
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1))
            next_generation.append(mutate(child2))
        
        population = next_generation

    # Find the best individual in the final population
    fitness_scores = [fitness(individual) for individual in population]
    best_individual = population[fitness_scores.index(max(fitness_scores))]
    best_features = [i for i, bit in enumerate(best_individual) if bit == 1]

    print(f"Best feature subset: {best_features}")
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
    X, y = load_and_preprocess_data()
    best_features = genetic_algorithm_feature_selection(X, y)
    X_selected = X.iloc[:, best_features]
    results = train_and_evaluate_model(X_selected, y)
    print("Neural network evaluation with GA-selected features completed successfully!")