# Configuración para Smart Shopping Cart

# Configuración de la aplicación Flask
SECRET_KEY = 'tu_clave_secreta_cambiar_en_produccion'
SQLALCHEMY_DATABASE_URI = 'sqlite:///smart_cart.db'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Configuración de archivos
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Configuración de detección
DETECTION_THRESHOLD = 0.5
YOLO_MODEL_PATH = 'yolov8n.pt'

# Configuración de cámara
DEFAULT_CAMERA_INDEX = 0

# Configuración de reportes
REPORTS_FOLDER = 'reports'
DAILY_REPORT_TIME = '23:59'  # Hora para generar reportes diarios
WEEKLY_REPORT_DAY = 'sunday'  # Día para generar reportes semanales

# Configuración de inventario
DEFAULT_MIN_STOCK = 5
LOW_STOCK_THRESHOLD = 5

# Configuración de precios sugeridos
DEFAULT_PRODUCT_PRICE = 10.00

# Configuración de la interfaz
ITEMS_PER_PAGE = 20
AUTO_REFRESH_INTERVAL = 5000  # milisegundos

# Configuración de logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'smart_cart.log'

# Configuración de desarrollo
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

