
# Tamizaje mediante machine learning para detección prenatal de cardiopatías congénitas

## Descripción

El script, 'train_delfos_exp.py', se utiliza para entrenar y evaluar el desempeño de un modelo utilizando el método Delfos. Al ejecutar este código, obtendrá métricas cruciales durante el entrenamiento, incluida la función objetivo (loss) y la precisión (accuracy) tanto en el conjunto de entrenamiento como en el conjunto de validación. Al finalizar el entrenamiento, se proporcionarán los grados de Sensibilidad, Especificidad, y los valores de VPP (Valor Predictivo Positivo), VPN (Valor Predictivo Negativo), FPr (Frecuencia de Falsos Positivos) y FNr (Frecuencia de Falsos Negativos).

## Requisitos Previos

Antes de ejecutar este script, asegúrese de tener los siguientes requisitos instalados:

Python (versión recomendada: 3.9.16)
pytorch
numpy
pandas
scikit-learn
Las bibliotecas necesarias pueden instalarse mediante el comando:
```bash
conda env create -f environment.yml':
```

## Cómo Usar

1. Configuración del Entorno:
- Asegúrese de que todos los requisitos previos estén instalados.
- Clone el repositorio si aún no lo ha hecho.
2. Ejecución del Script:
- Abra una terminal y navegue hasta el directorio que contiene train_delfos_exp.py.
- Ejecute el script usando el siguiente comando:
``` python
python train_delfos_exp.py
```
3. Resultados durante el Entrenamiento:
- Durante el entrenamiento, verá las métricas en tiempo real, que incluyen la función objetivo (loss) y la precisión (accuracy) en los conjuntos de entrenamiento y validación.
4. Resultados al Finalizar el Entrenamiento:
- Al finalizar el entrenamiento, se imprimirán en la terminal los resultados finales, que incluyen:
    - Sensibilidad
    - Especificidad
    - VPP (Valor Predictivo Positivo)
    - VPN (Valor Predictivo Negativo)
    - FPr (Frecuencia de Falsos Positivos)
    - FNr (Frecuencia de Falsos Negativos)


¡Gracias por utilizar train_delfos_exp.py! ¡Esperamos que tenga una experiencia de entrenamiento exitosa!