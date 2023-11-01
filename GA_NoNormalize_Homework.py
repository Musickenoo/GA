import numpy as np

# Load data from a file
def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', dtype=str)
    data = data[:, 1:]
    return data

#นำไฟส์มาทำ 10% cross validation โดยแบ่งข้อมูล 90% ออกมาเป็น train_data และ ข้อมูลอีก 10% ออกมาเป็น test_data เป็น 10 ช่วง
def split_data_into_segments(data, num_segments):
    segment_size = len(data) // num_segments

    test_data = []
    train_data = []

    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size

        test = data[start:end]
        train = np.concatenate([data[:start], data[end:]])

        test_data.append(test)
        train_data.append(train)

    return test_data, train_data

# Convert 'M' to 1 and 'B' to 0
def change_parameter(data):
    mapping = {'M': 1, 'B': 0}
    data[:, 0] = np.vectorize(mapping.get)(data[:, 0])
    return data

# Convert data to float and normalize it
def preprocess_data(data):
    data = change_parameter(data)
    
    # Convert to float
    data = data.astype(float)
    
    # Separate input and output
    input_data = data[:, 1:]
    output_data = data[:, 0]
    
    return input_data, output_data

# Get the min and max values from the dataset
def get_min_max(data):
    flattened_data = data.flatten()
    max_val = np.max(flattened_data)
    min_val = np.min(flattened_data)
    return min_val, max_val

# Normalize the data to be in the range [0, 1]
def normalize_data(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def inverse_data(data, min_val, max_val):
    # Perform inverse min-max scaling to denormalize the data
    original_data = data * (max_val - min_val) + min_val
    return original_data

def back_data(data):
    # คำนวณค่าเฉลี่ย
    avg_value = np.mean(data)

    # สร้าง Numpy array เพื่อเก็บผลลัพธ์
    result = np.empty(data.shape, dtype=str)

    # วนลูปผ่านข้อมูลและกำหนดค่า M หรือ B ตามเงื่อนไข
    for i in range(data.shape[0]):
        if data[i] > avg_value:
            result[i] = 'M'
        else:
            result[i] = 'B'

    return result

def calculate_error(actual, predict):
    if len(actual) != len(predict):
        raise ValueError("Actual and Predict must have the same length")

    error_count = 0
    total_samples = len(actual)

    for i in range(total_samples):
        if actual[i] != predict[i]:
            error_count += 1

    error_rate = error_count / total_samples
    accuracy = 1 - error_rate  # ความแม่นยำคือ 1 ลบออกจากความคลาดเคลื่อน

    return error_rate, accuracy*100
        
# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward propagation in the MLP
def forward_propagation(input_data, w_input_to_hidden, w_hidden_to_output):
    hidden = sigmoid(np.dot(w_input_to_hidden, input_data.T))
    output = sigmoid(np.dot(w_hidden_to_output, hidden))
    return hidden, output

# Training the MLP using Genetic Algorithms
def train_mlp_with_ga(input_data,output_data, hidden_size, num_generations, population_size, mutation_rate):

    
    input_size = input_data.shape[1]
    output_size = 1

    w_input_to_hidden = np.random.randn(hidden_size, input_size)
    w_hidden_to_output = np.random.randn(output_size, hidden_size)

    # Define your fitness function based on Mean Squared Error
    def fitness_function(weights, input_data, output_data, w_input_to_hidden, w_hidden_to_output):
        w_input_to_hidden = weights[:w_input_to_hidden.size].reshape(w_input_to_hidden.shape)
        w_hidden_to_output = weights[w_input_to_hidden.size:].reshape(w_hidden_to_output.shape)

        error = 0
        for i in range(input_data.shape[0]):
            hidden, output = forward_propagation(input_data[i], w_input_to_hidden, w_hidden_to_output)
            error += np.mean((output_data[i] - output) ** 2)
            #print(np.mean(error))
        return error
    
    for generation in range(num_generations):
        # Initialize the population with random weights
        population = np.random.randn(population_size, w_input_to_hidden.size + w_hidden_to_output.size)
     
        # Evaluate the fitness of each individual in the population
        fitness_scores = np.array([fitness_function(individual, input_data, output_data, w_input_to_hidden, w_hidden_to_output) 
                                   for individual in population])
        #print(fitness_scores)
        
        # Select the top-performing individuals to be parents
        parents = population[np.argsort(fitness_scores)[:int(0.2 * population_size)]]
        # Create a new population through crossover and mutation
        new_population = []

        while len(new_population) < population_size:
            parent_indices = np.random.choice(len(parents), size=2, replace=False)
            parent1 = parents[parent_indices[0]]
            parent2 = parents[parent_indices[1]]
            crossover_point = np.random.randint(parent1.size)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child += mutation_rate * np.random.randn(child.size)
            new_population.append(child)

        # Update the population
        population = np.array(new_population)
        
    # Find the best weights in the final population
    best_weights = population[np.argmin(fitness_scores)]
    
    #print(best_weights)
    # Train the MLP with the best weights
    w_input_to_hidden = best_weights[:w_input_to_hidden.size].reshape(w_input_to_hidden.shape)
    w_hidden_to_output = best_weights[w_input_to_hidden.size:].reshape(w_hidden_to_output.shape)
    
    return w_input_to_hidden, w_hidden_to_output

def calculate_accuracy(actual, predicted):
        # คำนวณความคลาดเคลื่อนร้อยละ
            errors = np.abs((actual - predicted) / actual) * 100

        # คำนวณค่า Accuracy โดยหาค่าเฉลี่ยของความถูกต้อง
            Accuracy = 100 - np.mean(errors)
        
            return Accuracy
        

if __name__ == "__main__":
    data_file = "WDBC.txt"
    
    input_size = 30
    hidden_size = 30 # สามารถกำหนดเองได้
    output_size = 1
    
    num_generations = 10# สามารถกำหนดเองได้
    population_size = 100# สามารถกำหนดเองได้
    mutation_rate = 0.5# สามารถกำหนดเองได้
    K_segments = 10
    
    print(f"Hidden node = {hidden_size} ")   
    for i in range(K_segments):
        
        print(f"segment = {i+1} num_generations = {num_generations} population_size ={population_size} mutation_rate = {mutation_rate}")   
        
        data = load_data(data_file)
        
        train_data, test_data = split_data_into_segments(data, K_segments)
        
        input_traindata, output_traindata = preprocess_data(train_data[i])
        
        w_input_to_hidden, w_hidden_to_output = train_mlp_with_ga(input_traindata,output_traindata, hidden_size,
                                                                  num_generations, population_size, mutation_rate)
          
        input_testdata, output_actual_normalize = preprocess_data(test_data[i])
        
        _,output_predict_normalize = forward_propagation(input_testdata, w_input_to_hidden, w_hidden_to_output)
        
        output_actual = output_actual_normalize.reshape(-1, 1)
        output_predict= output_predict_normalize.T
        
        output_actual = back_data(output_actual)
        output_predict = back_data(output_predict)
        #print(output_actual.shape)
        error_rate,Accuracy = calculate_error(output_actual,output_predict)
        correct_predictions = [actual == predict for actual, predict in zip(output_actual,output_predict)]

        # แสดงผลลัพธ์
        print("Actual Output\tPredict Output")
        #for actual, predict, is_correct in zip(output_actual, output_predict, correct_predictions):
            #if is_correct:
                #print(f"{' ' * 12}{actual}\t{' ' * 10}{predict}")
            #else:
                #print(f"{actual}\t{predict}")
                      
        print(f"************error_rate = {error_rate}  **************")
        print(f"************Accuracy = {Accuracy} % **************")
        
       
        
      
        
        
        

