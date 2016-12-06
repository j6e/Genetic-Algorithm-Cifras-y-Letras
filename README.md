# Cifras y Letras - Genetic Algorithm
Version 0.2

#How to run
Python 3.4+ required
```
usage: cifrasyletrasGA.py [-h] [--verbose]
                          [NUMBERS] [OBJECTIVE] [POPULATION] [GENERATIONS]
Cifras y Letras GA.

positional arguments:
  NUMBERS      Number of numbers to operate. Default 6
  OBJECTIVE    Upper bound of the objective. Default 999
  POPULATION   Size of the population. Default 50
  GENERATIONS  Number of generation. Default 1000

optional arguments:
  -h, --help   show this help message and exit
  --verbose    Verbose.
```

##Motivation
This code was first developed as an assigment for an AI related class.
The program aims to solve a simplification of the cifras challenge at the spanish TV show ["Cifras y Letras"](https://es.wikipedia.org/wiki/Cifras_y_letras) 
It was later enhanced to allow variations of the main problem.

##To-Do
* Make the numbers part of the DNA so they can be permuted, coming closer to solving the actual challenge.
