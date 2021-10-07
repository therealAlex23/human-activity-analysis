import numpy as np
#Dimensao da matriz
a = np.array([[1,2],[2,3]])
(l,c) = a.shape

#Meter numa linha só
b= a.flatten()

#Redimensionar matriz
a=np.array([[1,2,3,4]])
b=a.reshape((2,2))

#Mostrar partes de Matriz
b = [1,2,3,4]
#print(b[::])
#print(b[::2])

#Copiar uma matriz
a = np.array([[1,2],[3,4]])
#b = a --> como se fosse um ponteiro em c
b=a.copy() # Mantem uma copia para si mesmo
a[0,0] = 10
a[1,1] = b[0,0]

#Multiplicacao de matrizes
a = np.array([[1,2],[3,4]])
b=a.copy()
c = np.dot(a,b) # <=> a*b

#Somatorio mediante a orientação dos eixos --> array.func(axis = 0/1)
# axis = 1 --> orientação em linha
# axis = 0 --> orientação em coluna
b = a.sum(0)
b = a.sum(1)

#Situações com uma condição aplicad
print(a)
print(a>1)
print(a[a>1])
