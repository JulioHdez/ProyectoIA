#!/usr/bin/env python3
"""
Script de inicialización para Smart Shopping Cart
Configura la base de datos y descarga el modelo YOLO
"""

import os
import sys
import subprocess
import sqlite3
from pathlib import Path

def check_python_version():
    """Verificar que la versión de Python sea compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detectado")
    return True

def install_requirements():
    """Instalar dependencias desde requirements.txt"""
    print("\n📦 Instalando dependencias...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al instalar dependencias: {e}")
        return False

def create_directories():
    """Crear directorios necesarios"""
    print("\n📁 Creando directorios...")
    directories = ['uploads', 'reports', 'static', 'templates']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directorio '{directory}' creado/verificado")

def setup_database():
    """Configurar la base de datos SQLite"""
    print("\n🗄️ Configurando base de datos...")
    try:
        # Importar después de instalar dependencias
        from app import app, db
        
        with app.app_context():
            db.create_all()
            print("✅ Base de datos inicializada")
            
            # Crear algunos productos de ejemplo
            from models import Product, Inventory
            
            # Verificar si ya existen productos
            if Product.query.count() == 0:
                print("📝 Agregando productos de ejemplo...")
                
                sample_products = [
                    {"name": "Manzana", "price": 2.50, "category": "Frutas", "quantity": 50},
                    {"name": "Plátano", "price": 1.80, "category": "Frutas", "quantity": 30},
                    {"name": "Botella de Agua", "price": 5.00, "category": "Bebidas", "quantity": 25},
                    {"name": "Pan", "price": 3.50, "category": "Panadería", "quantity": 20},
                    {"name": "Leche", "price": 4.20, "category": "Lácteos", "quantity": 15},
                ]
                
                for product_data in sample_products:
                    product = Product(
                        name=product_data["name"],
                        price=product_data["price"],
                        category=product_data["category"],
                        description=f"Producto de ejemplo: {product_data['name']}"
                    )
                    db.session.add(product)
                    db.session.flush()
                    
                    inventory = Inventory(
                        product_id=product.id,
                        quantity=product_data["quantity"],
                        min_stock=5
                    )
                    db.session.add(inventory)
                
                db.session.commit()
                print("✅ Productos de ejemplo agregados")
            else:
                print("ℹ️ La base de datos ya contiene productos")
                
    except Exception as e:
        print(f"❌ Error al configurar la base de datos: {e}")
        return False
    
    return True

def download_yolo_model():
    """Descargar modelo YOLO"""
    print("\n🤖 Descargando modelo YOLO...")
    try:
        from ultralytics import YOLO
        
        # Descargar modelo YOLOv8n (nano) - más ligero
        model = YOLO('yolov8n.pt')
        print("✅ Modelo YOLO descargado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error al descargar modelo YOLO: {e}")
        print("ℹ️ El modelo se descargará automáticamente en el primer uso")
        return False

def create_startup_script():
    """Crear script de inicio"""
    print("\n🚀 Creando script de inicio...")
    
    if os.name == 'nt':  # Windows
        startup_script = """@echo off
echo Iniciando Smart Shopping Cart...
python app.py
pause
"""
        with open('start.bat', 'w') as f:
            f.write(startup_script)
        print("✅ Script de inicio creado: start.bat")
    else:  # Linux/Mac
        startup_script = """#!/bin/bash
echo "Iniciando Smart Shopping Cart..."
python3 app.py
"""
        with open('start.sh', 'w') as f:
            f.write(startup_script)
        os.chmod('start.sh', 0o755)
        print("✅ Script de inicio creado: start.sh")

def main():
    """Función principal de inicialización"""
    print("🎯 Smart Shopping Cart - Configuración Inicial")
    print("=" * 50)
    
    # Verificar Python
    if not check_python_version():
        return False
    
    # Instalar dependencias
    if not install_requirements():
        return False
    
    # Crear directorios
    create_directories()
    
    # Configurar base de datos
    if not setup_database():
        return False
    
    # Descargar modelo YOLO
    download_yolo_model()
    
    # Crear script de inicio
    create_startup_script()
    
    print("\n" + "=" * 50)
    print("🎉 ¡Configuración completada exitosamente!")
    print("\n📋 Próximos pasos:")
    print("1. Ejecuta la aplicación:")
    if os.name == 'nt':
        print("   - Windows: Ejecuta 'start.bat' o 'python app.py'")
    else:
        print("   - Linux/Mac: Ejecuta './start.sh' o 'python3 app.py'")
    print("2. Abre tu navegador en: http://localhost:5000")
    print("3. ¡Comienza a usar tu carrito inteligente!")
    
    print("\n💡 Consejos:")
    print("- Asegúrate de tener una cámara web conectada")
    print("- Los productos detectados se pueden agregar automáticamente")
    print("- Revisa el inventario regularmente para evitar stock bajo")
    print("- Genera reportes semanales para análisis de ventas")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ La configuración falló. Revisa los errores anteriores.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Configuración cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1)



