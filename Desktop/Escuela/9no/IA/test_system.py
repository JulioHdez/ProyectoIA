#!/usr/bin/env python3
"""
Script de prueba para Smart Shopping Cart
Verifica que todos los componentes funcionen correctamente
"""

import sys
import os
import cv2
import sqlite3
from pathlib import Path

def test_imports():
    """Probar que todas las dependencias se importen correctamente"""
    print("🔍 Probando importaciones...")
    
    try:
        import flask
        print("✅ Flask importado correctamente")
    except ImportError as e:
        print(f"❌ Error al importar Flask: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV importado correctamente")
    except ImportError as e:
        print(f"❌ Error al importar OpenCV: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ YOLO importado correctamente")
    except ImportError as e:
        print(f"❌ Error al importar YOLO: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy importado correctamente")
    except ImportError as e:
        print(f"❌ Error al importar NumPy: {e}")
        return False
    
    return True

def test_camera():
    """Probar acceso a la cámara web"""
    print("\n📹 Probando acceso a la cámara...")
    
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Cámara accesible y funcionando")
                cap.release()
                return True
            else:
                print("❌ No se pudo leer frame de la cámara")
                cap.release()
                return False
        else:
            print("❌ No se pudo abrir la cámara")
            return False
    except Exception as e:
        print(f"❌ Error al probar la cámara: {e}")
        return False

def test_database():
    """Probar conexión a la base de datos"""
    print("\n🗄️ Probando base de datos...")
    
    try:
        if os.path.exists('smart_cart.db'):
            conn = sqlite3.connect('smart_cart.db')
            cursor = conn.cursor()
            
            # Verificar tablas
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            expected_tables = ['products', 'inventory', 'sales', 'daily_reports', 'weekly_reports']
            existing_tables = [table[0] for table in tables]
            
            missing_tables = set(expected_tables) - set(existing_tables)
            if missing_tables:
                print(f"❌ Faltan tablas: {missing_tables}")
                return False
            
            print("✅ Base de datos accesible y tablas presentes")
            
            # Verificar datos
            cursor.execute("SELECT COUNT(*) FROM products")
            product_count = cursor.fetchone()[0]
            print(f"✅ Productos en base de datos: {product_count}")
            
            conn.close()
            return True
        else:
            print("❌ Base de datos no encontrada. Ejecuta setup.py primero")
            return False
    except Exception as e:
        print(f"❌ Error al probar la base de datos: {e}")
        return False

def test_yolo_model():
    """Probar modelo YOLO"""
    print("\n🤖 Probando modelo YOLO...")
    
    try:
        from ultralytics import YOLO
        
        # Intentar cargar el modelo
        model = YOLO('yolov8n.pt')
        print("✅ Modelo YOLO cargado correctamente")
        
        # Probar detección con imagen de prueba
        import numpy as np
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = model(test_image)
        print("✅ Detección YOLO funcionando")
        
        return True
    except Exception as e:
        print(f"❌ Error al probar YOLO: {e}")
        return False

def test_directories():
    """Probar que todos los directorios existan"""
    print("\n📁 Probando directorios...")
    
    required_dirs = ['uploads', 'reports', 'static', 'templates']
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directorio '{directory}' existe")
        else:
            print(f"❌ Directorio '{directory}' no existe")
            return False
    
    return True

def test_flask_app():
    """Probar que la aplicación Flask se pueda importar"""
    print("\n🌐 Probando aplicación Flask...")
    
    try:
        # Cambiar al directorio del proyecto
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Importar la aplicación
        from app import app
        
        # Probar configuración
        with app.app_context():
            print("✅ Aplicación Flask importada correctamente")
            print(f"✅ Base de datos configurada: {app.config['SQLALCHEMY_DATABASE_URI']}")
        
        return True
    except Exception as e:
        print(f"❌ Error al probar Flask: {e}")
        return False

def main():
    """Función principal de pruebas"""
    print("🧪 Smart Shopping Cart - Pruebas del Sistema")
    print("=" * 50)
    
    tests = [
        ("Importaciones", test_imports),
        ("Directorios", test_directories),
        ("Base de Datos", test_database),
        ("Modelo YOLO", test_yolo_model),
        ("Cámara Web", test_camera),
        ("Aplicación Flask", test_flask_app),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"⚠️ Prueba '{test_name}' falló")
        except Exception as e:
            print(f"❌ Error en prueba '{test_name}': {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Resultados: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! El sistema está listo.")
        print("\n🚀 Para iniciar la aplicación:")
        print("   python app.py")
        print("   Luego abre: http://localhost:5000")
    else:
        print("⚠️ Algunas pruebas fallaron. Revisa los errores anteriores.")
        print("\n💡 Sugerencias:")
        print("- Ejecuta 'python setup.py' para configurar el sistema")
        print("- Verifica que todas las dependencias estén instaladas")
        print("- Asegúrate de tener una cámara web conectada")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Pruebas canceladas por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1)

