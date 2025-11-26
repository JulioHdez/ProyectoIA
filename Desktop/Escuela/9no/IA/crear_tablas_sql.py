"""
Script robusto para crear las tablas en SQL Server

Este script verifica la conexión y crea las tablas paso a paso.

Uso:
    python crear_tablas_sql.py
"""
from app import app, db
from models import (
    Product, Inventory, Sale, DailyReport, WeeklyReport,
    TrainingProduct, TrainingImage
)

def crear_tablas():
    """Crea todas las tablas en la base de datos con verificación"""
    with app.app_context():
        print("=" * 60)
        print("CREANDO TABLAS EN SQL SERVER")
        print("=" * 60)
        
        try:
            # Verificar conexión
            print("\n[1/3] Verificando conexión a la base de datos...")
            db.engine.connect()
            print("  [OK] Conexión exitosa")
            
            # Verificar que la base de datos existe
            print("\n[2/3] Verificando base de datos...")
            result = db.engine.execute(db.text("SELECT DB_NAME()")).fetchone()
            db_name = result[0] if result else "desconocida"
            print(f"  [OK] Conectado a: {db_name}")
            
            # Crear tablas
            print("\n[3/3] Creando tablas...")
            db.create_all()
            print("  [OK] Comando create_all() ejecutado")
            
            # Verificar que las tablas se crearon
            print("\n[VERIFICACIÓN] Verificando tablas creadas...")
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            expected_tables = [
                'products',
                'inventory', 
                'sales',
                'daily_reports',
                'weekly_reports',
                'training_products',
                'training_images'
            ]
            
            print(f"\n  Tablas encontradas en la base de datos: {len(tables)}")
            for table in tables:
                print(f"    - {table}")
            
            missing_tables = [t for t in expected_tables if t not in tables]
            if missing_tables:
                print(f"\n  [ADVERTENCIA] Faltan {len(missing_tables)} tabla(s):")
                for table in missing_tables:
                    print(f"    - {table}")
            else:
                print("\n  [OK] Todas las tablas están creadas correctamente")
            
            print("\n" + "=" * 60)
            print("PROCESO COMPLETADO")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n[ERROR] Error al crear tablas: {e}")
            print("\nDetalles del error:")
            import traceback
            traceback.print_exc()
            
            print("\n" + "=" * 60)
            print("SOLUCIÓN DE PROBLEMAS")
            print("=" * 60)
            print("\n1. Verifica que la base de datos 'smart_cart' existe")
            print("2. Verifica que tienes permisos para crear tablas")
            print("3. Verifica la conexión en config.py")
            print("4. Revisa los errores de arriba")

if __name__ == '__main__':
    crear_tablas()

