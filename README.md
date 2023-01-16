# CONVERT CELSIUS TO FAHRENHEIT APPLYING ARTIFICIAL INTELLIGENCE WITH PYTHON AND TENSORFLOW

Simple neural network applying machine learning to convert celsius to fahrenheit.

## Differences between regular programming and machine learning

### Regular programming

Normally, we programming algorithms to become inputs in outputs; so, we write the rules and logic to get it.

![](https://github.com/faidrn/convert-celsius-to-fahrenheit-applying-AI-with-python/blob/main/resources/images/regularProgramming.png)


### Machine learning

We have a list of inputs and outputs, but we don't know how turn that inputs in that outputs. That is, we don't know the algorithm that can do the conversion.

![](https://github.com/faidrn/convert-celsius-to-fahrenheit-applying-AI-with-python/blob/main/resources/images/machineLearning.png)



## Convert Celsius to Fahrenheit 

![](https://github.com/faidrn/convert-celsius-to-fahrenheit-applying-AI-with-python/blob/main/resources/images/formula.png)

### Scenarios

#### Regular programming

Using regular programming, we would write a function like this:

```python
def function(C):
    F = C * 1.8 + 32
    return F
```

### Machine learning

Assuming that the algorithm is not known, and we have the input and output data

![](https://github.com/faidrn/convert-celsius-to-fahrenheit-applying-AI-with-python/blob/main/resources/images/valores.png)

A neural network is applied, looking for the model to learn the algorithm by itself.

![](https://github.com/faidrn/convert-celsius-to-fahrenheit-applying-AI-with-python/blob/main/resources/images/neuralNetwork.png)


**Note:** for this model the simplest neural network that can exist is used

![](https://github.com/faidrn/convert-celsius-to-fahrenheit-applying-AI-with-python/blob/main/resources/images/simpleNetwork.png)


Neural networks follow some rules like:

They are separated by layers, each layer can have one or more neurons.

![](https://github.com/faidrn/convert-celsius-to-fahrenheit-applying-AI-with-python/blob/main/resources/images/capasYNeuronas.png)

Any network has at least one input and one output layer; in more complex networks you can have more layers, but that is not the case with this model.

![](https://github.com/faidrn/convert-celsius-to-fahrenheit-applying-AI-with-python/blob/main/resources/images/inout.png)

The neurons are connected with connections. 

![](https://github.com/faidrn/convert-celsius-to-fahrenheit-applying-AI-with-python/blob/main/resources/images/conexiones.png)

For this model there will only be one connection.

![](https://github.com/faidrn/convert-celsius-to-fahrenheit-applying-AI-with-python/blob/main/resources/images/conexionSimple.png)

Each of the connections has a weight (numerical value) assigned, which represents the importance of the connection of the neurons

![](https://github.com/faidrn/convert-celsius-to-fahrenheit-applying-AI-with-python/blob/main/resources/images/pesosConexiones.png)

Every neuron except the input layer has a bias (numerical value), which controls how predisposed the neuron is to fire a 1 or a 0, regardless of the weights.


#### Applying these concepts in this model

The first neuron will have degrees Celsius, which will multiply with the weight of the connection and will reach the next neuron, where the bias will be added, and that will be the result (degrees Fahrenheit).

![](https://github.com/faidrn/convert-celsius-to-fahrenheit-applying-AI-with-python/blob/main/resources/images/proceso.png)

**Note:** It is important to keep in mind that when starting the weight and bias values start randomly.


#### Example

The network starts with weight of 1.5 and a bias of 4.

In this example, 15 degrees Celsius will be used.

We want the network to tell us what 15 degrees Celsius is converted to Fahrenheit, or at least what it thinks it is.

We put 15 degrees Celsius in the input neuron, this value is multiplied by 1.5, which is the weight of the connection, the result is 22.5

22.5 goes into the next neuron, and adds to the bias, which is 4 and the final value is 26.5

Therefore, at the moment the neural network predicts that 15 degrees Celsius will be 26.5 degrees Fahrenheit.

![](https://github.com/faidrn/convert-celsius-to-fahrenheit-applying-AI-with-python/blob/main/resources/images/ejemplo.png)

We use google engine to check the answer.

![](https://github.com/faidrn/convert-celsius-to-fahrenheit-applying-AI-with-python/blob/main/resources/images/google.png)


The result is not correct, so this is the point where machine learning works its magic.

**How can machine learning do it?**

We have to use enough input and output examples

**How to get the network to adjust its weights and biases in order to make the most accurate predictions possible?**

In order to make the most accurate predictions possible, the network will take all the input data and for each one it will make a prediction. Since it is randomly initialized, you will not get correct predictions; but depending on how bad the results were, it will adjust the weights and biases.



#### Now the code
