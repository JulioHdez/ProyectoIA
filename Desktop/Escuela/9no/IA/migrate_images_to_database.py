"""
Script para migrar imágenes de carpetas a base de datos

Uso:
    python migrate_images_to_database.py
"""
import os
import cv2
from app import app, db
from models import TrainingProduct, TrainingImage
from custom_trainer import CustomProductTrainer

def migrate_images_to_database():
    """Migra todas las imágenes de carpetas a la base de datos"""
    with app.app_context():
        # Crear tablas si no existen
        db.create_all()
        
        print("=" * 60)
        print("MIGRACIÓN DE IMÁGENES A BASE DE DATOS")
        print("=" * 60)
        
        # Inicializar entrenador de archivos
        file_trainer = CustomProductTrainer()
        
        # Obtener productos desde archivos
        custom_products = file_trainer.custom_products
        
        if not custom_products:
            print("\nNo se encontraron productos en las carpetas")
            return
        
        print(f"\nEncontrados {len(custom_products)} productos en carpetas")
        
        total_images = 0
        migrated_products = 0
        
        for product_name, product_info in custom_products.items():
            print(f"\nProcesando: {product_name}")
            
            # Verificar si el producto ya existe en BD
            existing_product = TrainingProduct.query.filter_by(name=product_name).first()
            if existing_product:
                print(f"  Producto '{product_name}' ya existe en BD, omitiendo...")
                continue
            
            # Crear producto en BD
            product = TrainingProduct(name=product_name)
            db.session.add(product)
            db.session.flush()  # Para obtener el ID
            
            # Obtener rutas de imágenes
            image_paths = product_info.get('image_paths', [])
            
            if not image_paths:
                print(f"  No se encontraron imágenes para '{product_name}'")
                continue
            
            # Migrar cada imagen
            migrated_count = 0
            for image_path in image_paths:
                if os.path.exists(image_path):
                    try:
                        # Leer imagen
                        img = cv2.imread(image_path)
                        if img is None:
                            print(f"    Error: No se pudo leer {image_path}")
                            continue
                        
                        # Codificar a JPEG
                        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        image_bytes = buffer.tobytes()
                        
                        # Obtener nombre de archivo
                        filename = os.path.basename(image_path)
                        
                        # Verificar si la imagen ya existe
                        existing_image = TrainingImage.query.filter_by(
                            product_id=product.id,
                            filename=filename
                        ).first()
                        
                        if not existing_image:
                            # Crear registro en BD
                            training_image = TrainingImage(
                                product_id=product.id,
                                filename=filename,
                                image_data=image_bytes,
                                image_format='JPEG',
                                file_size=len(image_bytes)
                            )
                            db.session.add(training_image)
                            migrated_count += 1
                    except Exception as e:
                        print(f"    Error procesando {image_path}: {e}")
                        continue
            
            if migrated_count > 0:
                db.session.commit()
                print(f"  ✓ Migradas {migrated_count} imágenes de '{product_name}'")
                migrated_products += 1
                total_images += migrated_count
            else:
                db.session.rollback()
                print(f"  ✗ No se migraron imágenes de '{product_name}'")
        
        print("\n" + "=" * 60)
        print("MIGRACIÓN COMPLETADA")
        print("=" * 60)
        print(f"Productos migrados: {migrated_products}")
        print(f"Imágenes migradas: {total_images}")
        print("\nNota: Las imágenes originales en carpetas NO se eliminan automáticamente.")
        print("Puedes eliminarlas manualmente después de verificar que todo funciona correctamente.")

if __name__ == '__main__':
    migrate_images_to_database()

