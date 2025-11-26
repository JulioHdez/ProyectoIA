"""
Entrenador que usa base de datos para almacenar imágenes en lugar de archivos
"""
import cv2
import numpy as np
import os
import json
import base64
from datetime import datetime
from models import db, TrainingProduct, TrainingImage

class DatabaseTrainer:
    """Clase para entrenar modelos usando imágenes almacenadas en base de datos"""
    
    def __init__(self):
        """Inicializa el entrenador con base de datos"""
        self.training_data_dir = 'training_data'
        self.custom_model_path = 'custom_products.pt'
        self.is_training = False
        
        # Crear directorios necesarios para entrenamiento temporal
        self._create_directories()
    
    def _create_directories(self):
        """Crea los directorios necesarios para el entrenamiento temporal"""
        directories = [
            self.training_data_dir,
            os.path.join(self.training_data_dir, 'images'),
            os.path.join(self.training_data_dir, 'labels'),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def add_product_for_training(self, product_name, images, labels=None):
        """
        Agrega un producto para entrenamiento (almacena en BD)
        
        Args:
            product_name (str): Nombre del producto
            images (list): Lista de imágenes (numpy arrays o base64 strings)
            labels (list): Lista de etiquetas (opcional)
        """
        try:
            print(f"[DatabaseTrainer] Agregando producto '{product_name}' con {len(images)} imágenes")
            
            # Buscar o crear producto en BD
            product = TrainingProduct.query.filter_by(name=product_name).first()
            if not product:
                print(f"[DatabaseTrainer] Creando nuevo producto '{product_name}' en BD")
                product = TrainingProduct(name=product_name)
                db.session.add(product)
                db.session.flush()  # Obtener el ID sin commit
            else:
                print(f"[DatabaseTrainer] Producto '{product_name}' ya existe (ID: {product.id})")
            
            saved_count = 0
            # Procesar y guardar imágenes
            for i, image in enumerate(images):
                try:
                    # Convertir imagen a bytes
                    if isinstance(image, str):
                        # Es base64 string
                        if image.startswith('data:image'):
                            # Remover prefijo data:image/jpeg;base64,
                            image_data = image.split(',')[1]
                        else:
                            image_data = image
                        
                        # Decodificar base64 directamente a bytes
                        image_bytes = base64.b64decode(image_data)
                        
                    elif isinstance(image, np.ndarray):
                        # Es numpy array - codificar a JPEG
                        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        image_bytes = buffer.tobytes()
                    else:
                        print(f"[DatabaseTrainer] Tipo de imagen no soportado: {type(image)}")
                        continue
                    
                    # Crear nombre de archivo único
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{product_name.replace(' ', '_')}_{i}_{timestamp}.jpg"
                    
                    # Verificar si la imagen ya existe (por nombre)
                    existing_image = TrainingImage.query.filter_by(
                        product_id=product.id,
                        filename=filename
                    ).first()
                    
                    if not existing_image:
                        # Crear registro de imagen en BD
                        training_image = TrainingImage(
                            product_id=product.id,
                            filename=filename,
                            image_data=image_bytes,
                            image_format='JPEG',
                            file_size=len(image_bytes)
                        )
                        db.session.add(training_image)
                        saved_count += 1
                        print(f"[DatabaseTrainer] Imagen {i+1}/{len(images)} guardada: {filename} ({len(image_bytes)} bytes)")
                    else:
                        print(f"[DatabaseTrainer] Imagen {filename} ya existe, omitiendo")
                        
                except Exception as img_error:
                    print(f"[DatabaseTrainer] Error procesando imagen {i}: {img_error}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Commit todas las imágenes de una vez
            db.session.commit()
            print(f"[DatabaseTrainer] ✓ Producto '{product_name}' agregado con {saved_count} imágenes en BD")
            return True
            
        except Exception as e:
            db.session.rollback()
            print(f"[DatabaseTrainer] ✗ Error al agregar producto: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def capture_images_from_web(self, product_name, images, num_images):
        """
        Captura imágenes desde la interfaz web y las guarda en BD
        
        Args:
            product_name (str): Nombre del producto
            images (list): Lista de imágenes en base64
            num_images (int): Número de imágenes capturadas
        """
        return self.add_product_for_training(product_name, images)
    
    def get_custom_products(self):
        """Obtiene la lista de productos personalizados desde BD"""
        products = TrainingProduct.query.all()
        return {p.name: {
            'name': p.name,
            'image_count': p.get_image_count(),
            'created_at': p.created_at.isoformat(),
            'last_updated': p.last_updated.isoformat()
        } for p in products}
    
    def get_product_images(self, product_name):
        """Obtiene los nombres de archivo de las imágenes de un producto"""
        product = TrainingProduct.query.filter_by(name=product_name).first()
        if not product:
            return []
        
        images = TrainingImage.query.filter_by(product_id=product.id).all()
        return [img.filename for img in images]
    
    def get_product_image_data(self, product_name, filename):
        """Obtiene los datos binarios de una imagen específica"""
        product = TrainingProduct.query.filter_by(name=product_name).first()
        if not product:
            return None
        
        image = TrainingImage.query.filter_by(
            product_id=product.id,
            filename=filename
        ).first()
        
        if image:
            return image.image_data
        return None
    
    def prepare_training_data(self, for_tensorflow=False):
        """
        Prepara los datos para el entrenamiento (extrae de BD a archivos temporales)
        
        Args:
            for_tensorflow (bool): Si True, crea estructura para TensorFlow. Si False, para YOLO.
        
        Returns:
            bool: True si se prepararon los datos correctamente
        """
        try:
            products = TrainingProduct.query.all()
            if not products:
                print("No hay productos personalizados para entrenar")
                return False
            
            print("Preparando datos de entrenamiento desde BD...")
            
            import shutil
            
            if for_tensorflow:
                # Estructura para TensorFlow: custom_products/Producto/images/
                base_dir = 'custom_products'
                if os.path.exists(base_dir):
                    shutil.rmtree(base_dir)
                os.makedirs(base_dir, exist_ok=True)
                
                # Extraer imágenes de BD a estructura de TensorFlow
                for product in products:
                    product_dir = os.path.join(base_dir, product.name)
                    images_dir = os.path.join(product_dir, 'images')
                    os.makedirs(images_dir, exist_ok=True)
                    
                    images = TrainingImage.query.filter_by(product_id=product.id).all()
                    print(f"  {product.name}: {len(images)} imágenes")
                    
                    for img in images:
                        image_path = os.path.join(images_dir, img.filename)
                        with open(image_path, 'wb') as f:
                            f.write(img.image_data)
                
                print(f"Datos de TensorFlow preparados: {len(products)} productos")
                return True
            else:
                # Estructura para YOLO: training_data/images/ y training_data/labels/
                images_dir = os.path.join(self.training_data_dir, 'images')
                labels_dir = os.path.join(self.training_data_dir, 'labels')
                
                # Limpiar directorios temporales
                for directory in [images_dir, labels_dir]:
                    if os.path.exists(directory):
                        shutil.rmtree(directory)
                    os.makedirs(directory)
                
                # Extraer imágenes de BD a archivos temporales
                for product_id, product in enumerate(products):
                    images = TrainingImage.query.filter_by(product_id=product.id).all()
                    
                    for img in images:
                        # Guardar imagen temporalmente
                        image_path = os.path.join(images_dir, img.filename)
                        with open(image_path, 'wb') as f:
                            f.write(img.image_data)
                        
                        # Crear etiqueta YOLO (asumiendo que el objeto ocupa todo el frame)
                        label_filename = img.filename.replace('.jpg', '.txt')
                        label_path = os.path.join(labels_dir, label_filename)
                        
                        # Formato YOLO: class_id center_x center_y width height
                        with open(label_path, 'w') as f:
                            f.write(f"{product_id} 0.5 0.5 1.0 1.0\n")
                
                print(f"Datos de entrenamiento YOLO preparados: {len(products)} productos")
                return True
            
        except Exception as e:
            print(f"Error preparando datos: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def delete_product(self, product_name):
        """Elimina un producto y todas sus imágenes de la BD"""
        try:
            product = TrainingProduct.query.filter_by(name=product_name).first()
            if product:
                # Las imágenes se eliminarán automáticamente por cascade
                db.session.delete(product)
                db.session.commit()
                print(f"Producto '{product_name}' eliminado de BD")
                return True
            else:
                print(f"Producto '{product_name}' no encontrado")
                return False
        except Exception as e:
            db.session.rollback()
            print(f"Error eliminando producto: {e}")
            return False
    
    def get_training_status(self):
        """Obtiene el estado del entrenamiento"""
        product_count = TrainingProduct.query.count()
        products = TrainingProduct.query.all()
        
        return {
            'is_training': self.is_training,
            'custom_products_count': product_count,
            'custom_model_exists': os.path.exists(self.custom_model_path),
            'products': [p.name for p in products]
        }
    
    def get_detailed_products(self):
        """Obtiene información detallada de los productos personalizados"""
        products = TrainingProduct.query.all()
        return [{
            'name': p.name,
            'image_count': p.get_image_count(),
            'created_at': p.created_at.isoformat(),
            'last_updated': p.last_updated.isoformat()
        } for p in products]
    
    def _create_data_yaml(self, yaml_path):
        """Crea el archivo data.yaml necesario para YOLO"""
        products = TrainingProduct.query.all()
        
        # Crear estructura de clases
        names = [p.name for p in products]
        nc = len(names)
        
        yaml_content = f"""path: {os.path.abspath(self.training_data_dir)}
train: images
val: images
test: images

nc: {nc}
names: {names}
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
    
    def train_custom_model(self, epochs=50, batch_size=16):
        """
        Entrena el modelo personalizado usando YOLO
        
        Args:
            epochs (int): Número de épocas de entrenamiento
            batch_size (int): Tamaño del batch
            
        Returns:
            bool: True si el entrenamiento fue exitoso
        """
        if self.is_training:
            print("Ya hay un entrenamiento en progreso")
            return False
        
        # Preparar datos desde BD
        if not self.prepare_training_data(for_tensorflow=False):
            return False
        
        try:
            self.is_training = True
            print("Iniciando entrenamiento del modelo personalizado desde BD...")
            
            # Importar YOLO
            from ultralytics import YOLO
            
            # Cargar modelo base
            model = YOLO('yolov8n.pt')
            
            # Configurar entrenamiento
            data_yaml = os.path.join(self.training_data_dir, 'data.yaml')
            self._create_data_yaml(data_yaml)
            
            # Entrenar modelo
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=640,
                device='cpu',  # Usar CPU para compatibilidad
                project='custom_training',
                name='product_detection'
            )
            
            # Guardar modelo entrenado
            trained_model_path = os.path.join('custom_training', 'product_detection', 'weights', 'best.pt')
            if os.path.exists(trained_model_path):
                import shutil
                shutil.copy2(trained_model_path, self.custom_model_path)
                print(f"Modelo personalizado guardado en: {self.custom_model_path}")
                return True
            else:
                print("Error: No se pudo encontrar el modelo entrenado")
                return False
                
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.is_training = False

