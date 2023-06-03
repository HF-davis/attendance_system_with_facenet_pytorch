# Attendance_system_with_facenet_pytorch
sistema de reconocimiento facial

# Sistema de texto a voz
Nuestro sistema emite una señal de audio personalizado con el nombre de la persona que esta haciendo uso de nuestro sistema, sea para el registro de entrada o salida. Cuando la señal de audio es emitida la persona sabrá que 
el sistema ya lo reconocio mencionando su nombre. El script <strong>text_spech.py</strong> se encarga de lo mencionado.
# Base de datos
El sistema para guardar un registro de asistencia hace uso de una base de datos, en donde se almacena el nombre, la hora de entrada, hora de salida, foto de entrada, foto de salida y un estado binario de entrada o salida de la persona
el script que describe la conexión de una base de datos con nuestro sistema es <strong>database.py</strong>.
# Server
Podemos correr el servidor con el siguiente comando
```python
python server.py
```
El servidor es el encargado de manejar el sistema, el servidor se encargar de registrar en la base de datos, emitir la señal de voz para que el usuario
sepa que el sistema lo ha reconocido, tomar las fotos necesarias para el entrenamiento del modelo (esto desde el front-end), entrenar el modelo de manera directa (esto desde el front-end), etc.
