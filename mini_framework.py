import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons=8):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.w_matrix = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.bias = np.zeros(n_neurons)
    def forward(self, inputs):
        self.inputs = inputs 
        XW = np.dot(inputs, self.w_matrix)
        Z = XW + self.bias
        return Z
    def backward(self, dinput):
        self.dw = np.dot(self.inputs.T, dinput)
        self.db = np.sum(dinput, axis=0, keepdims=True)
        self.dinputs = np.dot(dinput, self.w_matrix.T)
    
class Activation_ReLU:
    def forward(self, z_input):
        self.input = z_input
        A = np.maximum(0, z_input)
        return A
    def backward(self, dinput):
        self.dinput = dinput.copy()
        self.dinput[self.input <=0] = 0

class Activation_Softmax:
    def forward(self, A_input):
        euler_list = np.exp(np.array(A_input))
        try:
            euler_total = np.sum(euler_list, axis=1, keepdims=True)
        except np.exceptions.AxisError:
            euler_total = np.sum(euler_list, axis=0, keepdims=True)
        return euler_list/euler_total
    
class Loss_CategoricalCrossEntropy:
    def __init__(self):
        self.errors = []
    def forward(self, y_pred, y_true):
        n_y_pred = np.clip(y_pred, 0.000001, 0.999999)
        mtx = n_y_pred * y_true
        try:
            mtx = np.sum(mtx, axis=1, keepdims=True)
        except:
            mtx = np.sum(mtx, axis=0, keepdims=True)            
        mtx =  -np.log(mtx)
        self.errors.append(np.mean(mtx))
        return np.mean(mtx)
    
class Activation_Softmax_Loss_Loss_CategoricalCrossEntropy:
    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        y_hat = np.clip(y_pred, 0.000001, 0.999999)
        self.dinputs = (y_hat - y_true) / samples
        return self.dinputs

class Optimizer_SGD:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.w_matrix = layer.w_matrix - (self.learning_rate * layer.dw)
        layer.bias = layer.bias - (self.learning_rate * layer.db)

class Simple_Training_Loop:
    def __init__(self, n_hidden_layers=1, n_inputs=None, epochs=100, eta=0.01):
        self.n_hidden = n_hidden_layers
        self.n_inputs = n_inputs
        self.eta = eta
        self.epochs = epochs
        hidden = [Layer_Dense(n_inputs), ]
        hidden_relu = [Activation_ReLU(), ]
        for i in range(n_hidden_layers - 1):
            hidden.append(Layer_Dense(hidden[i-1].n_neurons))
            hidden_relu.append(Activation_ReLU())
        self.hidden_relu = hidden_relu
        self.hidden = hidden

    def hidden_layer_neuron_set(self, hidden_layer_number, n_neurons):
        if hidden_layer_number == 0:
            self.hidden[0] = Layer_Dense(self.hidden[0].n_inputs, n_neurons)    

        else:
            self.hidden[hidden_layer_number] = Layer_Dense(self.hidden[hidden_layer_number-1].n_neurons, n_neurons)
        try:
            self.hidden[hidden_layer_number+1] = Layer_Dense(self.hidden[hidden_layer_number].n_neurons, self.hidden[hidden_layer_number+1].n_neurons)
        except:
            pass

    def loop (self, input_array, y_true_array):
        exit_loss = Activation_Softmax_Loss_Loss_CategoricalCrossEntropy()
        graf_loss = Loss_CategoricalCrossEntropy()
        optimizer = Optimizer_SGD(self.eta)
        softmax = Activation_Softmax()
        try:
            exit = Layer_Dense(self.hidden[-1].n_neurons, len(y_true_array[0][0]))
        except TypeError:
            exit = Layer_Dense(self.hidden[-1].n_neurons, 1)       
        next = 0
        y_axis = []
        for epochs in range(self.epochs):
            for batch in range(len(input_array)): 
                for layer in range(len(self.hidden)):
                    if layer == 0:
                        hidden_dense = self.hidden[0].forward(input_array[batch])
                        next = self.hidden_relu[0].forward(hidden_dense)
                    else:
                        hidden_dense = self.hidden[layer].forward(next)
                        next = self.hidden_relu[layer].forward(hidden_dense)
                e = exit.forward(next)
                sm = softmax.forward(e)
                exit_loss.backward(sm, y_true_array[batch])
                graf_loss.forward(sm, y_true_array[batch])

                exit.backward(exit_loss.dinputs)
                optimizer.update_params(exit)

                for layer in range(len(self.hidden)):
                    if layer == 0:
                        self.hidden_relu[-1].backward(exit.dinputs)
                        self.hidden[-1].backward(self.hidden_relu[-1].dinput)
                        optimizer.update_params(self.hidden[-1])
                    else:
                        self.hidden_relu[-(1+layer)].backward(self.hidden[-(layer)].dinputs)
                        self.hidden[-(1+layer)].backward(self.hidden_relu[-(1+layer)].dinput)
                        optimizer.update_params(self.hidden[-(1+layer)])
            y_axis.append(np.mean(graf_loss.errors))
            graf_loss.errors = []


        self.exit_layer = exit

        plt.figure(figsize=(14, 6))
        plt.plot(range(1, self.epochs + 1), y_axis, marker='o', linestyle='-', color='blue')
        plt.title('Mean Error per Training Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Error')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def test(self, input_array_test, y_true_array_test):
        graf_loss = Loss_CategoricalCrossEntropy()
        softmax = Activation_Softmax()
        
        for batch in range(len(input_array_test)):
            next_input = 0
            
            for layer in range(len(self.hidden)):
                if layer == 0:
                    hidden_dense = self.hidden[0].forward(input_array_test[batch])
                    next_input = self.hidden_relu[0].forward(hidden_dense)
                else:
                    hidden_dense = self.hidden[layer].forward(next_input)
                    next_input = self.hidden_relu[layer].forward(hidden_dense)
            
            e = self.exit_layer.forward(next_input)
            sm = softmax.forward(e)
            
            loss = graf_loss.forward(sm, y_true_array_test[batch])
            
            y_pred_classes = np.argmax(sm, axis=1)
            y_true_classes = np.argmax(y_true_array_test[batch], axis=1)
            
            acuracia = np.mean(y_pred_classes == y_true_classes)
            
            cm = pd.crosstab(y_true_classes, y_pred_classes, 
                             rownames=['Real'], colnames=['Predict'])
            
            print(f"\n--- TEST RESULT ---")
            print(f"Loss: {loss:.4f}")
            print(f"Accuracy: {acuracia * 100:.2f}%\n")
            print("Confusion Matrix:")
            print(cm)
            print("-" * 36)