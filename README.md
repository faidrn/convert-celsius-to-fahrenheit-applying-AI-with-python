# CONVERT CELSIUS TO FAHRENHEIT APPLYING ARTIFICIAL INTELLIGENCE WITH PYTHON AND TENSORFLOW

Simple neural network applying machine learning to convert celsius to fahrenheit.

## Differences between regular programming and machine learning

### Regular programming

Normally, we programming algorithms to become inputs in outputs; so, we write the rules and logic to get it.

![](https://github.com/faidrn/convert-celsius-to-fahrenheit-applying-AI-with-python/blob/main/resources/images/regularProgramming.png)


### Machine learning

We have a list of inputs and outputs, but we don't know how turn that inputs in that outputs. That is, we don't know the algorithm that can do the conversion.

![](https://github.com/faidrn/convert-celsius-to-fahrenheit-applying-AI-with-python/blob/main/resources/images/machineLearning.png)

=============

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