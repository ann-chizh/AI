import numpy as np

class Neuron:  
    def __init__(self, n_inputs, bias=0., weights=None):  
        self.b = bias
        if weights:
            self.ws = np.array(weights)
        else:
            self.ws = np.random.rand(n_inputs)

    def _f(self, x):  
        return max(x * 0.1, x)   

    def __call__(self, xs):  
        return self._f(xs @ self.ws + self.b) 

class Layer:
    def __init__(self, n_neurons, n_inputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def __call__(self, xs):
        return np.array([neuron(xs) for neuron in self.neurons])

class ANN:
    def __init__(self):
        self.input_layer = Layer(3, 3)
        self.hidden_layer1 = Layer(4, 3)  
        self.hidden_layer2 = Layer(4, 4) 
        self.output_layer = Layer(1, 4)  

    def __call__(self, xs):
        x = self.input_layer(xs)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.output_layer(x)
        return x

import networkx as nx
import matplotlib.pyplot as plt

def draw_ann():
    G = nx.DiGraph()
    
    for i in range(3):
        G.add_node(f'Input {i+1}', layer=0)
        
    for i in range(4):
        G.add_node(f'Hidden1 {i+1}', layer=1)
        
    for i in range(4):
        G.add_node(f'Hidden2 {i+1}', layer=2)
        
    G.add_node('Output', layer=3)
    
    for i in range(3):
        for j in range(4):
            G.add_edge(f'Input {i+1}', f'Hidden1 {j+1}')
            
    for i in range(4):
        for j in range(4):
            G.add_edge(f'Hidden1 {i+1}', f'Hidden2 {j+1}')
            
    
    for i in range(4):
        G.add_edge(f'Hidden2 {i+1}', 'Output')
    
    pos = nx.multipartite_layout(G, subset_key='layer')
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    plt.show()

draw_ann()
