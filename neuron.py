import tensorflow as tf # Libreria para IA
import numpy as np # Libreria para trabajar con arreglos numericos
# matplotlib => generación de gráficos en dos dimensiones, a partir de datos contenidos en listas o arrays
# matplotlib.pyplot => Pyplot is an API (Application Programming Interface), interface estilo MATLAB; 
# pyplot almacena el estado de un objeto cuando lo traza por primera vez.
import matplotlib.pyplot as plt # Usamos esta libreria para ver el resultado de la funcion de perdida


# Ejemplos que la red neuronal usara para aprender 

# Arreglo de numeros con las entradas en grados celsius
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)

# Array con los resultados en grados fahrenheit
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)


# Modelo de red neuronal
# Usamos keras ya que permite para especificar las capas de entrada y salida por separado, o especificar solo la capa de salida

# Creamos una capa del tipo Dense, estas capas son las que tienen conexiones desde cada neurona hacia todas las neuronas de la 
# siguiente capa
# units=1 => solo tiene una neurona de salida
# input_shape=[1] => indica que tiene una entrada con una neurona tambien (auto registra la capa de entrada con una neurona)
capa = tf.keras.layers.Dense(units=1, input_shape=[1])

# Estas capas estan volando, por lo que se debe usar un modelo de keras para darle las capas y poder trabajar con el

# Modelo secuencial
modelo = tf.keras.Sequential([capa])


# La compilacion prepara el modelo para ser entrenado
# Le indicamos como queremos que procese las matematicas para que aprendar de una mejor forma

# Le indicamos 2 propiedades:

# Optimizador (optimizer), usamos Adam, el cual le permite a la red saber como ajustar los pesos y sesgos de manera eficiente para q aprenda 
# y no desaprenda, es decir, q poco a poco vaya mejorando en lugar de ir empeorando

# Taza de aprendizaje (0.1), este numero le indica al modelo q tanto ajustar los pesos y sesgos; si se pone un numero muy pequeño los ira 
# ajustando muy poco, la red aprendera muy lento, pero si el numero es muy grande quizas se pase del numero esperado y no pueda hacer 
# cambios lo suficientemente finos para llegar a la mejor opcion

# Funcion de perdida (loss), para esta funcion usamos 'mean_squared_error', la cual considera que una poca cantidad de errores grandes es 
# peor que una gran cantidad de errores pequeños
modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1), 
    loss = 'mean_squared_error'
)


# Entrenamiento
# Se usa la funcion fit para entrenar el modelo
print('Comenzando entrenamiento...')

# modelo.fit(datos de entrada, resultados esperados, epochs= numero de vueltas que debe intentarlo, verbose= 'False evita que 
# imprima datos no deseados')
# La cantidad de vueltas le da tiempo al modelo para q se optimice lo mas posible
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)

print('Modelo entrenado!')


# Mostramos el resultado de la funcion de perdida
# La funcion de perdida nos dice q tan mal estan los resultados de la red en cada vuelta que dió
plt.xlabel('# de vuelta')
plt.ylabel('Magnitud de pérdida')
plt.plot(historial.history['loss'])


# Prediccion
print('Predicción')
resultado = modelo.predict([100.0])
print(f'El resultado es {str(resultado)} fahrenheit!')


# Imprimir las variables internas del modelo
print("Variables internas del modelo")
print("[peso (primer array), sesgo (segundo array)]")
print(capa.get_weights())