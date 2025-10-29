# Smart Shopping Cart - Sistema de Carrito Inteligente

## Descripción
Sistema de carrito de compras inteligente que utiliza visión por computadora para detectar productos automáticamente, gestionar inventario y generar reportes de ventas.

## Características
- 🎥 Detección automática de productos con IA
- 📦 Gestión de inventario en tiempo real
- 💰 Seguimiento de ventas diarias
- 📊 Reportes semanales automáticos
- 🌐 Interfaz web multiplataforma

## Instalación

1. Clona el repositorio:
```bash
git clone <tu-repositorio>
cd smart-shopping-cart
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecuta la aplicación:
```bash
python app.py
```

4. Abre tu navegador en: http://localhost:5000

## Estructura del Proyecto
```
smart-shopping-cart/
├── app.py                 # Aplicación principal Flask
├── models.py             # Modelos de base de datos
├── object_detector.py    # Detección de objetos con IA
├── camera_handler.py     # Manejo de cámara web
├── templates/            # Plantillas HTML
├── static/              # Archivos estáticos (CSS, JS)
├── uploads/             # Imágenes subidas
└── reports/             # Reportes generados
```

## Uso
1. Configura tu cámara web
2. Agrega productos al inventario
3. Inicia la detección automática
4. Visualiza ventas y reportes

## Tecnologías Utilizadas
- Python 3.8+
- Flask (Framework web)
- OpenCV (Visión por computadora)
- YOLO (Detección de objetos)
- SQLite (Base de datos)
- HTML/CSS/JavaScript (Frontend)

