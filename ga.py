import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# Load Data
def load_data(filename):
    return pd.read_csv(filename, header=None)

def load_labels(filename):
    return pd.read_csv(filename, header=None)

# Initialize Population
def initialize_population(population_size, num_features):
    return np.random.randint(2, size=(population_size, num_features))


def cosine_similarity_custom(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def evaluate_fitness(population, training_data, training_labels, validation_data, validation_labels):
    fitness_scores = []
    for chromosome in population:
        selected_features = training_data.iloc[:, chromosome == 1]
        class_centroids = []
        for class_label in range(7):
            class_instances = selected_features[training_labels[0] == class_label]
            class_centroid = class_instances.mean(axis=0).values.reshape(1, -1)
            class_centroids.append(class_centroid)
        validation_data_selected = validation_data.iloc[:, chromosome == 1]
        similarity_scores = []
        for instance in range(validation_data_selected.shape[0]):
            instance_scores = []
            for class_centroid in class_centroids:
                similarity_score = cosine_similarity_custom(class_centroid, validation_data_selected.iloc[instance])
                instance_scores.append(similarity_score)
            similarity_scores.append(instance_scores)
        print(similarity_scores)
        similarity_scores = np.array(similarity_scores)
        predicted_labels = np.argmax(similarity_scores, axis=2)
        predicted_labels = np.squeeze(predicted_labels)  # Convert to 1D array
        accuracy = np.mean(predicted_labels == validation_labels[0])
        if np.isnan(accuracy):
            accuracy = 0  # Replace NaN with 0
        fitness_scores.append(accuracy)
    return np.array(fitness_scores)


def selection(population, fitness_scores, selection_percentage):
    if np.all(np.isnan(fitness_scores)) or np.all(fitness_scores == 0):
        # Handle the case when all fitness scores are zero or NaN.
        selected_indices = np.random.choice(len(population), size=int(len(population) * selection_percentage), replace=False)
    else:
        selected_indices = np.random.choice(len(population), size=int(len(population) * selection_percentage), replace=False, p=np.nan_to_num(fitness_scores) / np.sum(np.nan_to_num(fitness_scores)))
    return selected_indices

# Crossover
def crossover(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i+1]
        crossover_point = np.random.randint(len(parent1))
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        offspring.extend([child1, child2])
    return np.array(offspring)

# Mutation
def mutation(offspring_population, mutation_rate):
    mutated_offspring_population = offspring_population.copy()
    for chromosome in mutated_offspring_population:
        for i in range(len(chromosome)):
            if np.random.uniform() < mutation_rate:
                chromosome[i] = 1 - chromosome[i]  # Flip bit
    return mutated_offspring_population

# Termination
def termination_condition_met(generation, max_generations):
    return generation >= max_generations

# Main Genetic Algorithm
def genetic_algorithm(train_data_file, train_label_file, validate_data_file, validate_label_file,
                      population_size=200, num_generations=50, selection_percentage=0.3, mutation_rate=0.1):
    # Load Data
    training_data = load_data(train_data_file)
    training_labels = load_labels(train_label_file)
    validation_data = load_data(validate_data_file)
    validation_labels = load_labels(validate_label_file)

    num_features = training_data.shape[1]

    # Initialize Population
    population = initialize_population(population_size, num_features)

    # Main Loop
    for generation in range(num_generations):
        # Evaluate Fitness
        print(generation)
        if generation == 0:
            continue
        fitness_scores = evaluate_fitness(population, training_data, training_labels, validation_data, validation_labels)

        # Selection
        selected_indices = selection(population, fitness_scores, selection_percentage)

        # Crossover
        offspring_population = crossover(population[selected_indices])

        # Mutation
        mutated_offspring_population = mutation(offspring_population, mutation_rate)

        # Replacement
        population[selected_indices] = mutated_offspring_population

        # Termination
        if termination_condition_met(generation, num_generations):
            break

    # Select Best Solution
    best_solution_index = np.argmax(fitness_scores)
    best_solution = population[best_solution_index]
    print(fitness_scores)
    return best_solution, fitness_scores[best_solution_index]

# Example Usage
best_features, best_fitness_score = genetic_algorithm("train.csv", "label_train.csv", "validate.csv", "label_validate.csv")
print("Best selected features:", best_features)
print("Best fitness score:", best_fitness_score)