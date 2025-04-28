import random
import math
import tkinter as tk
from tkinter import ttk, messagebox

class NeuralCell:
    def __init__(self, bias_value=None):
        self.bias_value = bias_value if bias_value is not None else random.uniform(-0.1, 0.1)
        self.weight_values = []
        self.output_value = 0
        self.input_sum_value = 0
  
    def activate_neuron(self, input_values):
        self.input_sum_value = sum(weight * input_value for weight, input_value in zip(self.weight_values, input_values)) + self.bias_value
        self.output_value = self.hyperbolic_tangent_activation(self.input_sum_value)
        return self.output_value
  
    def hyperbolic_tangent_activation(self, x_value):
        x_clipped = max(min(x_value, 10), -10)
        exponential_value = math.exp(2 * x_clipped)
        return (exponential_value - 1) / (exponential_value + 1)
  
    def hyperbolic_tangent_derivative(self, output_value):
        return 1.0 - output_value ** 2
  
    def softmax_normalization(self, value_list):
        max_value = max(value_list)
        exponential_values = [math.exp(value - max_value) for value in value_list]
        sum_exponentials = sum(exponential_values)
        return [value / sum_exponentials for value in exponential_values]

class RecurrentNetworkLayer:
    def __init__(self, number_of_neurons, number_of_inputs):
        self.neurons = [NeuralCell() for _ in range(number_of_neurons)]
        
        for neuron in self.neurons:
            neuron.weight_values = [random.uniform(-0.1, 0.1) for _ in range(number_of_inputs)]
        
        self.recurrent_weight_matrix = [[random.uniform(-0.1, 0.1) for _ in range(number_of_neurons)] 
                                      for _ in range(number_of_neurons)]
        
        self.current_outputs = [0] * number_of_neurons
        self.previous_outputs = [0] * number_of_neurons
  
    def forward_propagation(self, input_values):
        self.previous_outputs = self.current_outputs.copy()
        
        new_outputs = []
        for neuron_index, neuron in enumerate(self.neurons):
            input_activation = sum(weight * input_value for weight, input_value in zip(neuron.weight_values, input_values))
            
            recurrent_activation = sum(recurrent_weight * previous_output 
                                    for recurrent_weight, previous_output 
                                    in zip(self.recurrent_weight_matrix[neuron_index], self.previous_outputs))
            
            neuron.input_sum_value = input_activation + recurrent_activation + neuron.bias_value
            neuron.output_value = neuron.hyperbolic_tangent_activation(neuron.input_sum_value)
            new_outputs.append(neuron.output_value)
        
        self.current_outputs = new_outputs
        return new_outputs

class OutputPredictionLayer:
    def __init__(self, number_of_neurons, number_of_inputs):
        self.neurons = [NeuralCell() for _ in range(number_of_neurons)]
        
        for neuron in self.neurons:
            neuron.weight_values = [random.uniform(-0.1, 0.1) for _ in range(number_of_inputs)]
  
    def forward_propagation(self, input_values):
        output_values = [neuron.activate_neuron(input_values) for neuron in self.neurons]
        
        probability_distribution = self.neurons[0].softmax_normalization([neuron.input_sum_value for neuron in self.neurons])
        
        for neuron_index, neuron in enumerate(self.neurons):
            neuron.output_value = probability_distribution[neuron_index]
        
        return probability_distribution

class RecurrentNeuralNetwork:
    def __init__(self):
        self.word_to_index_dictionary = {'delta': 0, 'university': 1, 'is': 2, 'my': 3, 'dream': 4}
        self.index_to_word_dictionary = {0: 'delta', 1: 'university', 2: 'is', 3: 'my', 4: 'dream'}
        self.vocabulary_size = len(self.word_to_index_dictionary)
        self.hidden_layer_size = 10
        self.hidden_layer = RecurrentNetworkLayer(self.hidden_layer_size, self.vocabulary_size)
        self.output_layer = OutputPredictionLayer(self.vocabulary_size, self.hidden_layer_size)
        self.learning_rate_parameter = 0.01
        self.train_network()

    def train_network(self, number_of_epochs=1000):
        training_inputs = [[1 if index == word_index else 0 for word_index in range(self.vocabulary_size)] 
                         for index in range(3)]
        
        target_outputs = [[1 if index == word_index else 0 for word_index in range(self.vocabulary_size)] 
                        for index in [3, 4, 2]]
        
        for epoch in range(number_of_epochs):
            input_sequence, hidden_states, predicted_probabilities = self.forward_pass(training_inputs)
            gradient_values = self.backward_pass(input_sequence, hidden_states, predicted_probabilities, target_outputs)
            self.update_network_parameters(*gradient_values)

    def forward_pass(self, input_sequence):
        input_values = []
        hidden_state_values = []
        output_probabilities = []
        
        self.hidden_layer.previous_outputs = [0] * self.hidden_layer_size
        
        for time_step in range(len(input_sequence)):
            current_input = input_sequence[time_step]
            input_values.append(current_input)
            
            hidden_state = self.hidden_layer.forward_propagation(current_input)
            hidden_state_values.append(hidden_state)
            
            output_probability = self.output_layer.forward_propagation(hidden_state)
            output_probabilities.append(output_probability)
        
        return input_values, hidden_state_values, output_probabilities

    def backward_pass(self, input_sequence, hidden_states, predicted_probabilities, target_outputs):
        input_weight_gradients = [[0 for _ in range(self.vocabulary_size)] for _ in range(self.hidden_layer_size)]
        recurrent_weight_gradients = [[0 for _ in range(self.hidden_layer_size)] for _ in range(self.hidden_layer_size)]
        output_weight_gradients = [[0 for _ in range(self.hidden_layer_size)] for _ in range(self.vocabulary_size)]
        hidden_bias_gradients = [0] * self.hidden_layer_size
        output_bias_gradients = [0] * self.vocabulary_size
        
        next_hidden_gradient = [0] * self.hidden_layer_size
        
        for time_step in reversed(range(len(input_sequence))):
            output_error = [predicted_probability - target_value 
                          for predicted_probability, target_value 
                          in zip(predicted_probabilities[time_step], target_outputs[time_step])]
            
            for output_index in range(self.vocabulary_size):
                for hidden_index in range(self.hidden_layer_size):
                    output_weight_gradients[output_index][hidden_index] += output_error[output_index] * hidden_states[time_step][hidden_index]
                output_bias_gradients[output_index] += output_error[output_index]
            
            hidden_error = [0] * self.hidden_layer_size
            for hidden_index in range(self.hidden_layer_size):
                for output_index in range(self.vocabulary_size):
                    hidden_error[hidden_index] += self.output_layer.neurons[output_index].weight_values[hidden_index] * output_error[output_index]
                hidden_error[hidden_index] += next_hidden_gradient[hidden_index]
            
            tanh_derivative_values = [hidden_error[hidden_index] * self.hidden_layer.neurons[hidden_index].hyperbolic_tangent_derivative(hidden_states[time_step][hidden_index]) 
                                     for hidden_index in range(self.hidden_layer_size)]
            
            for hidden_index in range(self.hidden_layer_size):
                hidden_bias_gradients[hidden_index] += tanh_derivative_values[hidden_index]
                
                for input_index in range(self.vocabulary_size):
                    input_weight_gradients[hidden_index][input_index] += tanh_derivative_values[hidden_index] * input_sequence[time_step][input_index]
                
                if time_step > 0:
                    for previous_hidden_index in range(self.hidden_layer_size):
                        recurrent_weight_gradients[hidden_index][previous_hidden_index] += tanh_derivative_values[hidden_index] * hidden_states[time_step-1][previous_hidden_index]
            
            next_hidden_gradient = [0] * self.hidden_layer_size
            for hidden_index in range(self.hidden_layer_size):
                for next_hidden_index in range(self.hidden_layer_size):
                    next_hidden_gradient[next_hidden_index] += tanh_derivative_values[hidden_index] * self.hidden_layer.recurrent_weight_matrix[hidden_index][next_hidden_index]

        return input_weight_gradients, recurrent_weight_gradients, output_weight_gradients, hidden_bias_gradients, output_bias_gradients

    def update_network_parameters(self, input_weight_gradients, recurrent_weight_gradients, output_weight_gradients, hidden_bias_gradients, output_bias_gradients):
        for hidden_index in range(self.hidden_layer_size):
            for input_index in range(self.vocabulary_size):
                self.hidden_layer.neurons[hidden_index].weight_values[input_index] -= self.learning_rate_parameter * input_weight_gradients[hidden_index][input_index]
        
        for hidden_index in range(self.hidden_layer_size):
            for previous_hidden_index in range(self.hidden_layer_size):
                self.hidden_layer.recurrent_weight_matrix[hidden_index][previous_hidden_index] -= self.learning_rate_parameter * recurrent_weight_gradients[hidden_index][previous_hidden_index]
        
        for output_index in range(self.vocabulary_size):
            for hidden_index in range(self.hidden_layer_size):
                self.output_layer.neurons[output_index].weight_values[hidden_index] -= self.learning_rate_parameter * output_weight_gradients[output_index][hidden_index]
        
        for hidden_index in range(self.hidden_layer_size):
            self.hidden_layer.neurons[hidden_index].bias_value -= self.learning_rate_parameter * hidden_bias_gradients[hidden_index]
        
        for output_index in range(self.vocabulary_size):
            self.output_layer.neurons[output_index].bias_value -= self.learning_rate_parameter * output_bias_gradients[output_index]

    def predict_next_word(self, words):
        word_indices = [self.word_to_index_dictionary[word] for word in words]
        input_vector = [1 if index in word_indices else 0 for index in range(self.vocabulary_size)]
        hidden_state = self.hidden_layer.forward_propagation(input_vector)
        output_probability = self.output_layer.forward_propagation(hidden_state)
        predicted_index = output_probability.index(max(output_probability))
        return self.index_to_word_dictionary[predicted_index]

class NeuralNetworkInterface:
    def __init__(self, root_window):
        self.root_window = root_window
        self.root_window.title("Neural Network Word Prediction System")
        self.root_window.geometry("600x400")
        self.root_window.configure(bg="#f0f2f5")
        
        self.neural_network = RecurrentNeuralNetwork()
        
        self.initialize_interface_components()
    
    def initialize_interface_components(self):
        style_configurator = ttk.Style()
        style_configurator.configure('MainFrame.TFrame', background="#f0f2f5")
        style_configurator.configure('TitleLabel.TLabel', background="#f0f2f5", font=('Arial', 16, 'bold'))
        style_configurator.configure('InputLabel.TLabel', background="#f0f2f5", font=('Arial', 10))
        style_configurator.configure('PredictButton.TButton', font=('Arial', 10, 'bold'), foreground="black", background="#4CAF50")
        style_configurator.configure('ResultFrame.TFrame', background="#ffffff", relief=tk.GROOVE, borderwidth=2)
        
        main_container = ttk.Frame(self.root_window, style='MainFrame.TFrame')
        main_container.pack(pady=25, padx=25, fill=tk.BOTH, expand=True)
        
        title_component = ttk.Label(main_container, 
                                  text="Delta University Word Prediction Engine", 
                                  style='TitleLabel.TLabel')
        title_component.pack(pady=(0, 20))
        
        input_container = ttk.Frame(main_container)
        input_container.pack(pady=15)
        
        input_label = ttk.Label(input_container, 
                              text="Enter three consecutive words:", 
                              style='InputLabel.TLabel')
        input_label.grid(row=0, column=0, columnspan=3, pady=5)
        
        self.first_word_entry = ttk.Entry(input_container, width=15, font=('Arial', 10))
        self.first_word_entry.grid(row=1, column=0, padx=7, pady=7)
        self.first_word_entry.insert(0, "delta")
        
        self.second_word_entry = ttk.Entry(input_container, width=15, font=('Arial', 10))
        self.second_word_entry.grid(row=1, column=1, padx=7, pady=7)
        self.second_word_entry.insert(0, "university")
        
        self.third_word_entry = ttk.Entry(input_container, width=15, font=('Arial', 10))
        self.third_word_entry.grid(row=1, column=2, padx=7, pady=7)
        self.third_word_entry.insert(0, "is")
        
        prediction_button = ttk.Button(main_container, 
                                     text="Prediction", 
                                     command=self.execute_prediction,
                                     style='PredictButton.TButton')
        prediction_button.pack(pady=15)
        
        self.result_display_frame = ttk.Frame(main_container, style='ResultFrame.TFrame')
        self.result_display_frame.pack(fill=tk.X, pady=15, ipady=10)
        
        result_label = ttk.Label(self.result_display_frame, 
                               text="Prediction Result:", 
                               font=('Arial', 10),
                               background="#ffffff")
        result_label.pack(side=tk.LEFT, padx=10)
        
        self.prediction_result_label = ttk.Label(self.result_display_frame, 
                                               text="", 
                                               font=('Arial', 10, 'bold'),
                                               foreground="#2196F3",
                                               background="#ffffff")
        self.prediction_result_label.pack(side=tk.LEFT)
        
        vocabulary_info = ttk.Label(main_container, 
                                  text="Available vocabulary: delta, university, is, my, dream",
                                  font=('Arial', 8),
                                  background="#f0f2f5")
        vocabulary_info.pack(side=tk.BOTTOM, pady=(20, 0))
    
    def execute_prediction(self):
        try:
            input_words = [
                self.first_word_entry.get().lower().strip(),
                self.second_word_entry.get().lower().strip(),
                self.third_word_entry.get().lower().strip()
            ]
            
            for word in input_words:
                if word not in self.neural_network.word_to_index_dictionary:
                    raise ValueError(f"Word '{word}' is not in the recognized vocabulary")
            
            predicted_word = self.neural_network.predict_next_word(input_words)
            self.prediction_result_label.config(text=f"'{' '.join(input_words)}' → {predicted_word}")
        except Exception as error:
            self.prediction_result_label.config(text=str(error), foreground="red")

if __name__ == "__main__":
    application_root = tk.Tk()
    application_interface = NeuralNetworkInterface(application_root)
    application_root.mainloop()