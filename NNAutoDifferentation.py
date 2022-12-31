# Auto differentation program from: https://towardsdatascience.com/build-your-own-automatic-differentiation-program-6ecd585eec2a

import numpy as np

class Graph():
    # Computational graph class.
    # Initilizes a global variable _g that describes the graph.
    # Each Graph consists of a set of
    #   1. operators
    #   2. variables
    #   3. constants
    #   4. placeholders
    
    def __init__(self):
        self.operators = set()
        self.constants = set()
        self.variables = set()
        self.placeholders = set()
        global _g
        _g = self
        
    def reset_counts(self, root):
        if hasattr(root, 'count'):
            root.count = 0
        else:
            for child in root.__subclasses__():
                self.reset_counts(child)
        
    def __enter__(self):
        return self

class Node:
    def __init__(self):
        pass
    
class Placeholder(Node):
    count = 0
    
    def __init__(self, name, dtype = float):
        _g.placeholders.add(self)
        self.value = None
        self.gradient = None
        self.name = f"Plc/{Placeholder.count}" if name is None else name
        Placeholder.count +=1
        
    def __repr__(self):
        return f"Placeholder: name:{self.name}, value:{self.value}"
    
    
class Constant(Node):
    count = 0
    
    def __init__(self, value, name = None):
        _g.constants.add(self)
        self._value = value
        self.gradient = None
        self.name = f"Const/{Constant.count}" if name is None else name
        Constant.count += 1
        
    def __repr__(self):
        return f"Constant: name:{self.name}, value:{self.value}"
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self):
        raise ValueError("Cannot reassign constant")
        self.value = None
        self.gradient = None
        # self.name = f"Plc/{Placeholder.count}" if name is None else name
        self.name = f"Plc/{Placeholder.count}"
        Placeholder.count += 1
        
    # def __repr__(self):
    #     return f"Placeholder: name:{self.name}, value:{self.value}"
    
class Variable(Node):
    count = 0
    
    def __init__(self, value, name = None):
        _g.variables.add(self)
        self.value = value
        self.gradient = None
        self.name = f"Var/{Variable.count}" if name is None else name
        Variable.count += 1
        
    def __repr__(self):
        return f"Variable: name:{self.name}, value:{self.value}"
    
class Operator(Node):
    def __init__(self, name = 'Operator'):
        _g.operators.add(self)
        self.value = None
        self.inputs = []
        self.gradient = None
        self.name = name
        
    def __repr__(self):
        return f"Operator: name:{self.name}"
    
class add(Operator):
    count = 0
    
    def __init__(self, a, b, name = None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f"add/{add.count}" if name is None else name
        add.count += 1
        
    def forward(self, a, b):
        return a + b
    
    def backward(self, a, b, dout):
        return dout, dout
    
class multiply(Operator):
    count = 0
    
    def __init__(self, a, b, name = None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f"mul/{multiply.count}" if name is None else name
        multiply.count += 1
        
    def forward(self, a, b):
        return a * b
    
    def backward(self, a, b, dout):
        return dout * b, dout * a
    
class divide(Operator):
    count = 0
    
    def __init__(self, a, b, name = None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f"div/{divide.count}" if name is None else name
        divide.count += 1

    def forward(self, a, b):
        return a / b
    
    def backward(self, a, b, dout):
        return dout / b, dout * a / np.power(b, 2)
    
class power(Operator):
    count = 0
    
    def __init__(self, a, b, name = None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f"pow/{power.count}" if name is None else name
        power.count += 1
        
    def forward(self, a, b):
        return np.power(a, b)
    
    def backward(self, a, b, dout):
        return dout * b * np.power(a, (b - 1)), dout * np.log(a) * np.power(a, b)
    
class matmul(Operator):
    count = 0
    
    def __init__(self, a, b, name = None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f"matmul/{matmul.count}" if name is None else name
        matmul.count += 1
        
    def forward(self, a, b):
        return a@b
    
    def backward(self, a, b, dout):
        return dout@b.T, a.T@dout
    
def node_wrapper(func, self, other):
    if isinstance(other, Node):
        return func(self, other)
    if isinstance(other, float) or isinstance(other, int):
        return func(self, Constant(other))
    
    raise TypeError("Incompatible types.")
    
Node.__add__ = lambda self, other: node_wrapper(add, self, other)
Node.__mul__ = lambda self, other: node_wrapper(multiply, self, other)
Node.__div__ = lambda self, other: node_wrapper(divide, self, other)
Node.__neg__ = lambda self: node_wrapper(multiply, self, Constant(-1))
Node.__pow__ = lambda self, other: node_wrapper(power, self, other)
Node.__matmul__ = lambda self, other: node_wrapper(matmul, self, other)


##############################
#####  Topological sort  #####
##############################
def topological_sort(head_node=None, graph=_g):
    """Performs topological sort of all nodes prior to and 
    including the head_node. 
    Args:
        graph: the computational graph. This is the global value by default
        head_node: last node in the forward pass. The "result" of the graph.
    Returns:
        a sorted array of graph nodes.
    """
    vis = set()
    ordering = []
    
    def _dfs(node):
        if node not in vis:
            vis.add(node)
            if isinstance(node, Operator):
                for input_node in node.inputs:
                    _dfs(input_node)
            ordering.append(node)
            
    if head_node is None:
        for node in graph.operators:
            _dfs(node)
    else:
        _dfs(head_node)
        
    return ordering
    
  
##############################
#####    Forward pass    #####
##############################    
def forward_pass(order, feed_dict={}):
    """ Performs the forward pass, returning the output of the graph.
    Args:
        order: a topologically sorted array of nodes
        feed_dict: a dictionary values for placeholders.
    Returns:
        1. the final result of the forward pass.
        2. directly edits the graph to fill in its current values.
    """
    for node in order:
        
        if isinstance(node, Placeholder):
            node.value = feed_dict[node.name]
                    
        elif isinstance(node, Operator):
            node.value = node.forward(*[prev_node.value for prev_node in node.inputs])

    return order[-1].value
    
##############################
#####    Backward pass   #####
##############################  
def backward_pass(order, target_node=None):
    """ Perform the backward pass to retrieve gradients.
    Args:
        order: a topologically sorted array of graph nodes.
               by default, this assigns the graident of the final node to 1
    Returns:
        gradients of nodes as listed in same order as input argument
    """
    vis = set()
    order[-1].gradient = 1
    for node in reversed(order):
        if isinstance(node, Operator):
            inputs = node.inputs
            grads = node.backward(*[x.value for x in inputs], dout=node.gradient)
            for inp, grad in zip(inputs, grads):
                if inp not in vis:
                    inp.gradient = grad
                else:
                    inp.gradient += grad
                vis.add(inp)
    return [node.gradient for node in order]