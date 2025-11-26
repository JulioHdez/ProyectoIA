"""
Script para verificar la conexión a la base de datos configurada

Uso:
    python verificar_conexion_db.py
"""
from app import app, db
from models import Product, Inventory, TrainingProduct, TrainingImage

def verificar_conexion():
    """Verifica la conexión y muestra información de la base de datos"""
    with app.app_context():
        print("=" * 60)
        print("VERIFICACIÓN DE CONEXIÓN A BASE DE DATOS")
        print("=" * 60)
        
        # Mostrar URI de conexión (sin contraseña)
        db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', 'No configurada')
        if 'password' in db_uri.lower() or '@' in db_uri:
            # Ocultar contraseña
            parts = db_uri.split('@')
            if len(parts) > 1:
                db_uri_display = parts[0].split(':')[0] + ':***@' + parts[1]
            else:
                db_uri_display = db_uri
        else:
            db_uri_display = db_uri
        
        print(f"\nURI de conexión: {db_uri_display}")
        
        # Detectar tipo de base de datos
        if 'sqlite' in db_uri.lower():
            print("Tipo: SQLite")
        elif 'mssql' in db_uri.lower() or 'sqlserver' in db_uri.lower():
            print("Tipo: SQL Server")
        else:
            print("Tipo: Desconocido")
        
        try:
            # Intentar conectar
            print("\n[1/3] Intentando conectar a la base de datos...")
            db.engine.connect()
            print("  ✓ Conexión exitosa")
            
            # Verificar tablas
            print("\n[2/3] Verificando tablas...")
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            print(f"  Tablas encontradas: {len(tables)}")
            for table in tables:
                print(f"    - {table}")
            
            # Contar registros
            print("\n[3/3] Contando registros...")
            
            if 'products' in tables:
                product_count = Product.query.count()
                print(f"  Productos: {product_count}")
            
            if 'inventory' in tables:
                inventory_count = Inventory.query.count()
                print(f"  Inventario: {inventory_count}")
            
            if 'training_products' in tables:
                training_product_count = TrainingProduct.query.count()
                print(f"  Productos de entrenamiento: {training_product_count}")
            
            if 'training_images' in tables:
                training_image_count = TrainingImage.query.count()
                print(f"  Imágenes de entrenamiento: {training_image_count}")
            
            print("\n" + "=" * 60)
            print("VERIFICACIÓN COMPLETADA")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n✗ Error al conectar: {e}")
            print("\nVerifica:")
            print("  1. Que la base de datos 'smart_cart' existe en SQL Server")
            print("  2. Que tienes permisos para acceder")
            print("  3. Que el driver ODBC está instalado correctamente")
            print("  4. Que SQL Server está corriendo")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    verificar_conexion()

