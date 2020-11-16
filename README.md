# Autoencoders-implementations

## Como ejecutar

En el archivo build.xml se encuentran los siguientes tasks de ant para compilar y ejecutar los ejercicios:
* `ant compile` para compilar todos los ejercicios.
* `ant jar` para generar los .jar de todos los ejercicios.
* `ant run-ej1-a` para ejecutar el ejercicio 1)a).
* `ant run-ej1-b` para ejecutar el ejercicio 1)b).
* `ant run-ej2` para ejecutar el ejercicio 2).
* `ant clean` para borrar la carpeta /build.

## Dependecias

Hay un script en python llamado latentSpaceGraph.py que utiliza la información generada a partir de la ejecución del ejercicio 1)a) para graficar la representación de los puntos en el espacio latente del autoencoder.
Las dependencias de python que se necesitan tener instaladas para poder graficar, son:
* matplotlib
* numpy
