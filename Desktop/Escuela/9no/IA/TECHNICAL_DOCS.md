# Smart Shopping Cart - Documentación Técnica

## Descripción General

Smart Shopping Cart es un sistema de carrito de compras inteligente que utiliza visión por computadora e inteligencia artificial para detectar productos automáticamente, gestionar inventario y generar reportes de ventas.

## Arquitectura del Sistema

### Componentes Principales

1. **Backend (Flask)**
   - `app.py`: Aplicación principal Flask
   - `models.py`: Modelos de base de datos SQLAlchemy
   - `object_detector.py`: Sistema de detección de objetos con YOLO
   - `config.py`: Configuración del sistema

2. **Frontend (HTML/CSS/JavaScript)**
   - `templates/`: Plantillas Jinja2
   - `static/`: Archivos CSS y JavaScript
   - Interfaz responsive con Bootstrap 5

3. **Base de Datos (SQLite)**
   - `smart_cart.db`: Base de datos SQLite
   - Modelos: Product, Inventory, Sale, DailyReport, WeeklyReport

4. **IA y Visión por Computadora**
   - YOLOv8 para detección de objetos
   - OpenCV para procesamiento de imágenes
   - Mapeo de clases COCO a productos comerciales

## Funcionalidades

### 1. Detección Automática de Productos
- **Tecnología**: YOLOv8 (You Only Look Once)
- **Entrada**: Stream de cámara web en tiempo real
- **Procesamiento**: Detección de objetos con bounding boxes
- **Salida**: Lista de productos detectados con confianza

```python
# Ejemplo de uso del detector
detector = ObjectDetector()
detections = detector.detect_objects(image)
```

### 2. Gestión de Inventario
- **CRUD completo** de productos
- **Control de stock** con alertas de stock bajo
- **Categorización** de productos
- **Códigos de barras** opcionales

### 3. Sistema de Ventas
- **Registro automático** de ventas por detección IA
- **Registro manual** de ventas
- **Seguimiento temporal** de transacciones
- **Cálculo automático** de totales

### 4. Reportes y Análisis
- **Reportes diarios** automáticos
- **Reportes semanales** generables
- **Estadísticas de ventas** en tiempo real
- **Exportación a CSV**

## Modelos de Datos

### Product
```python
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    price = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50))
    barcode = db.Column(db.String(50), unique=True)
    image_path = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
```

### Inventory
```python
class Inventory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'))
    quantity = db.Column(db.Integer, nullable=False, default=0)
    min_stock = db.Column(db.Integer, default=5)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
```

### Sale
```python
class Sale(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'))
    quantity = db.Column(db.Integer, nullable=False)
    unit_price = db.Column(db.Float, nullable=False)
    total_price = db.Column(db.Float, nullable=False)
    sale_date = db.Column(db.Date, default=date.today)
    sale_time = db.Column(db.DateTime, default=datetime.utcnow)
    detection_method = db.Column(db.String(20), default='manual')
```

## API Endpoints

### Productos
- `GET /products` - Lista de productos
- `POST /add_product` - Agregar producto
- `POST /update_inventory/<id>` - Actualizar inventario

### Cámara y Detección
- `GET /camera` - Interfaz de cámara
- `POST /start_detection` - Iniciar detección
- `POST /stop_detection` - Detener detección
- `GET /get_detection_results` - Obtener resultados
- `POST /add_detected_product` - Agregar producto detectado

### Ventas y Reportes
- `GET /sales` - Ventas del día
- `GET /reports` - Reportes
- `POST /generate_weekly_report` - Generar reporte semanal
- `GET /export_report/<id>` - Exportar reporte

## Configuración del Sistema

### Requisitos del Sistema
- **Python**: 3.8 o superior
- **Cámara web**: Para detección de productos
- **RAM**: Mínimo 4GB (recomendado 8GB)
- **Almacenamiento**: 2GB libres

### Dependencias Principales
```
flask==2.3.3
flask-sqlalchemy==3.0.5
opencv-python==4.8.1.78
ultralytics==8.0.196
pillow==10.0.1
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
```

### Instalación
1. Clonar el repositorio
2. Ejecutar `python setup.py`
3. Ejecutar `python test_system.py` (opcional)
4. Ejecutar `python app.py`
5. Abrir http://localhost:5000

## Flujo de Trabajo

### 1. Detección de Productos
```
Cámara Web → OpenCV → YOLO → Mapeo de Clases → Interfaz Web
```

### 2. Procesamiento de Ventas
```
Detección → Validación → Actualización Inventario → Registro Venta → Reporte
```

### 3. Generación de Reportes
```
Datos de Ventas → Agregación → Análisis → Visualización → Exportación
```

## Personalización

### Agregar Nuevos Productos
1. Modificar `product_mapping` en `object_detector.py`
2. Agregar precios sugeridos en `_get_suggested_price()`
3. Entrenar modelo personalizado (opcional)

### Modificar Interfaz
1. Editar plantillas en `templates/`
2. Modificar estilos en `static/style.css`
3. Agregar JavaScript personalizado

### Configurar Base de Datos
1. Modificar `SQLALCHEMY_DATABASE_URI` en `config.py`
2. Cambiar a PostgreSQL/MySQL para producción
3. Configurar migraciones con Flask-Migrate

## Rendimiento y Optimización

### Detección en Tiempo Real
- **FPS objetivo**: 10-15 FPS
- **Resolución**: 640x480 recomendada
- **Umbral de confianza**: 0.5 (configurable)

### Base de Datos
- **Índices**: En campos de búsqueda frecuente
- **Limpieza**: Archivos de log rotativos
- **Backup**: Automático diario

### Escalabilidad
- **Horizontal**: Múltiples instancias Flask
- **Vertical**: Más CPU/RAM para mejor rendimiento
- **Caché**: Redis para sesiones frecuentes

## Seguridad

### Medidas Implementadas
- **Validación de entrada**: Sanitización de datos
- **CSRF Protection**: Tokens en formularios
- **SQL Injection**: ORM SQLAlchemy
- **XSS Prevention**: Escapado automático en templates

### Recomendaciones de Producción
- Cambiar `SECRET_KEY` por valor seguro
- Usar HTTPS en producción
- Implementar autenticación de usuarios
- Configurar firewall y proxy reverso

## Troubleshooting

### Problemas Comunes

#### Cámara no funciona
```bash
# Verificar dispositivos de video
ls /dev/video*

# Probar con OpenCV
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

#### Modelo YOLO no carga
```bash
# Descargar manualmente
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

#### Base de datos corrupta
```bash
# Eliminar y recrear
rm smart_cart.db
python setup.py
```

### Logs y Debugging
- **Logs de aplicación**: `smart_cart.log`
- **Debug mode**: `DEBUG = True` en `config.py`
- **Errores de Flask**: Habilitar modo debug

## Roadmap Futuro

### Funcionalidades Planificadas
- [ ] Reconocimiento facial para usuarios
- [ ] Integración con sistemas de pago
- [ ] App móvil complementaria
- [ ] Análisis predictivo de inventario
- [ ] Integración con proveedores
- [ ] Dashboard de administración avanzado

### Mejoras Técnicas
- [ ] Microservicios con Docker
- [ ] API REST completa
- [ ] Tests automatizados
- [ ] CI/CD pipeline
- [ ] Monitoreo con Prometheus
- [ ] Cache distribuido con Redis

## Contribución

### Cómo Contribuir
1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

### Estándares de Código
- **PEP 8**: Estilo de código Python
- **Docstrings**: Documentación de funciones
- **Type hints**: Tipado estático
- **Tests**: Cobertura mínima 80%

## Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## Contacto

Para soporte técnico o consultas:
- **Email**: soporte@smartcart.com
- **GitHub Issues**: Para reportar bugs
- **Documentación**: Wiki del proyecto

