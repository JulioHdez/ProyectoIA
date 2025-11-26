# Smart Shopping Cart - Sistema de Detección de Productos con IA

## Descripción

Sistema de carrito de compras inteligente que utiliza inteligencia artificial para la detección y reconocimiento automático de productos mediante visión por computadora. El sistema integra tecnologías de detección de objetos (YOLO) y clasificación de imágenes (TensorFlow) para identificar productos en tiempo real.

## Tecnologías Utilizadas

### Backend
- **Flask 2.3.3**: Framework web para la API y servidor
- **SQLAlchemy 3.0.5**: ORM para gestión de base de datos
- **Flask-Migrate 4.0.5**: Migraciones de base de datos

### Inteligencia Artificial
- **YOLOv8 (Ultralytics 8.0.196)**: Modelo de detección de objetos en tiempo real
- **TensorFlow 2.13.0**: Framework para clasificación de imágenes y aprendizaje profundo
- **Keras**: API de alto nivel para construcción de modelos de deep learning
- **OpenCV 4.8.1.78**: Procesamiento de imágenes y visión por computadora

### Base de Datos
- **SQL Server**: Base de datos principal (recomendado)
- **SQLite**: Base de datos alternativa para desarrollo
- **pyodbc 5.0.1**: Driver para conexión a SQL Server

### Procesamiento de Datos
- **NumPy 1.24.3**: Operaciones numéricas y arrays multidimensionales
- **Pandas 2.0.3**: Análisis y manipulación de datos
- **scikit-learn 1.3.0**: Algoritmos de machine learning y preprocesamiento

### Visualización
- **Matplotlib 3.7.2**: Generación de gráficos y visualizaciones
- **Seaborn 0.12.2**: Visualización estadística avanzada

### Frontend
- **Bootstrap 5.1.3**: Framework CSS para interfaz responsive
- **JavaScript (ES6+)**: Interactividad y comunicación con API
- **HTML5/CSS3**: Estructura y estilos de la interfaz

## Requisitos del Sistema

### Software
- **Python 3.8 o superior**
- **SQL Server 2017 o superior** (recomendado) o SQLite
- **ODBC Driver 17 o 18 for SQL Server** (si usas SQL Server)
- **Navegador web moderno** (Chrome, Firefox, Edge)

### Hardware Recomendado
- **CPU**: Procesador multi-core (Intel i5 o equivalente)
- **RAM**: Mínimo 8GB (16GB recomendado para entrenamiento)
- **GPU**: Opcional pero recomendada para entrenamiento más rápido (NVIDIA con CUDA)
- **Cámara web**: Para captura de imágenes en tiempo real
- **Almacenamiento**: Mínimo 5GB libres

### Dependencias Python
Todas las dependencias están especificadas en `requirements.txt`. Instalación:

```bash
pip install -r requirements.txt
```

## Instalación y Configuración

### 1. Clonar o Descargar el Proyecto

```bash
cd ruta/del/proyecto
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar Base de Datos

#### Opción A: SQL Server (Recomendado)

1. **Instalar ODBC Driver**:
   - Descargar desde: https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
   - Instalar "ODBC Driver 17 for SQL Server" o superior

2. **Crear Base de Datos**:
   ```sql
   CREATE DATABASE smart_cart;
   GO
   ```

3. **Configurar conexión en `config.py`**:
   ```python
   # SQL Server con Windows Authentication (local)
   SQLALCHEMY_DATABASE_URI = 'mssql+pyodbc://localhost/smart_cart?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
   ```

4. **Verificar driver ODBC**:
   ```bash
   python verificar_odbc.py
   ```

5. **Crear tablas**:
   ```bash
   python crear_tablas_sql.py
   ```

#### Opción B: SQLite (Desarrollo)

En `config.py`, descomentar:
```python
SQLALCHEMY_DATABASE_URI = 'sqlite:///smart_cart.db'
```

### 4. Configurar Almacenamiento de Imágenes

En `config.py`:
```python
# Para almacenar en base de datos (SQL Server recomendado)
IMAGE_STORAGE_METHOD = 'database'

# Para almacenar en archivos (compatibilidad)
# IMAGE_STORAGE_METHOD = 'filesystem'
```

### 5. Verificar Configuración

```bash
# Verificar conexión a base de datos
python verificar_conexion_db.py

# Verificar drivers ODBC
python verificar_odbc.py
```

### 6. Iniciar la Aplicación

```bash
python app.py
```

La aplicación estará disponible en: `http://localhost:5000`

## Estructura del Proyecto

```
smart-cart/
├── app.py                      # Aplicación Flask principal
├── models.py                   # Modelos de base de datos
├── object_detector.py          # Detector YOLO
├── custom_trainer.py           # Entrenador para archivos
├── database_trainer.py         # Entrenador para base de datos
├── tensorflow_trainer.py       # Entrenador TensorFlow
├── config.py                   # Configuración del sistema
├── requirements.txt            # Dependencias Python
├── templates/                  # Plantillas HTML
│   ├── index.html
│   ├── products.html
│   ├── camera.html
│   ├── training.html
│   └── ...
├── static/                     # Archivos estáticos
│   ├── style.css
│   └── ...
└── README.md                   # Este archivo
```

## Funcionalidades Principales

### 1. Detección de Productos en Tiempo Real
- Detección automática mediante YOLOv8
- Visualización de bounding boxes
- Múltiples productos simultáneos

### 2. Gestión de Inventario
- CRUD completo de productos
- Control de stock
- Alertas de inventario bajo

### 3. Entrenamiento Personalizado
- **YOLO**: Detección de objetos con bounding boxes
- **TensorFlow**: Clasificación de imágenes completas
- Captura de imágenes desde cámara web
- Almacenamiento en base de datos

### 4. Sistema de Ventas
- Registro automático de ventas
- Cálculo de totales
- Reportes diarios y semanales

### 5. Almacenamiento de Imágenes
- Almacenamiento en SQL Server (VARBINARY)
- Migración automática desde archivos
- Servicio de imágenes desde base de datos

## Uso Básico

### Agregar Producto Manualmente
1. Ir a "Productos" → "Agregar Producto"
2. Completar formulario
3. Guardar

### Entrenar Modelo Personalizado
1. Ir a "Entrenamiento"
2. Ingresar nombre del producto
3. Capturar imágenes (mínimo 10 recomendado)
4. Seleccionar tipo de entrenamiento (YOLO o TensorFlow)
5. Configurar parámetros (épocas, batch size)
6. Iniciar entrenamiento

### Usar Detección en Tiempo Real
1. Ir a "Cámara IA"
2. Iniciar detección
3. Los productos detectados se mostrarán automáticamente

## Solución de Problemas

### Error: "Driver not found"
- Instalar ODBC Driver 17 o 18 for SQL Server
- Verificar con `python verificar_odbc.py`

### Error: "Cannot open database"
- Verificar que la base de datos `smart_cart` existe
- Verificar permisos de usuario
- Verificar conexión con `python verificar_conexion_db.py`

### Error: "Empty logs" en TensorFlow
- Asegurar tener suficientes imágenes (mínimo 10 por producto)
- El batch_size se ajusta automáticamente si es necesario

### Cámara no funciona
- Verificar permisos del navegador
- Verificar que la cámara no esté en uso por otra aplicación
- Probar en otro navegador

## Notas Técnicas

- **YOLO**: Utiliza modelo pre-entrenado `yolov8n.pt` (nano) para detección base
- **TensorFlow**: Modelos pre-entrenados disponibles: MobileNetV2, EfficientNetB0, ResNet50
- **Base de Datos**: Las imágenes se almacenan como VARBINARY(MAX) en SQL Server
- **Threading**: El entrenamiento se ejecuta en hilos separados para no bloquear la interfaz
- **Matplotlib**: Configurado con backend 'Agg' para evitar problemas con tkinter en threads

## Licencia

Este proyecto es de uso educativo y académico.

## Autor

Desarrollado como proyecto académico para sistema de detección de productos con inteligencia artificial.

---

**Versión**: 1.0  
**Última actualización**: Noviembre 2025
