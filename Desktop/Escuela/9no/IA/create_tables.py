"""
Script para crear las tablas en la base de datos manualmente

Uso:
    python create_tables.py
"""
from app import app, db
from models import (
    Product, Inventory, Sale, DailyReport, WeeklyReport,
    TrainingProduct, TrainingImage
)

def create_tables():
    """Crea todas las tablas en la base de datos"""
    with app.app_context():
        print("=" * 60)
        print("CREANDO TABLAS EN LA BASE DE DATOS")
        print("=" * 60)
        
        try:
            # Crear todas las tablas
            db.create_all()
            
            print("\n[OK] Tablas creadas exitosamente!")
            print("\nTablas creadas:")
            print("  - products")
            print("  - inventory")
            print("  - sales")
            print("  - daily_reports")
            print("  - weekly_reports")
            print("  - training_products")
            print("  - training_images")
            
            print("\n" + "=" * 60)
            print("LISTO - Puedes ejecutar la aplicación ahora")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n[ERROR] Error al crear tablas: {e}")
            print("\nVerifica:")
            print("  1. Que la base de datos 'smart_cart' existe")
            print("  2. Que tienes permisos para crear tablas")
            print("  3. Que la conexión a SQL Server es correcta")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    create_tables()

