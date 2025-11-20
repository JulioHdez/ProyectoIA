import cv2
import numpy as np
import os
import json
from datetime import datetime
from ultralytics import YOLO
import shutil
from pathlib import Path

class CustomProductTrainer:
    """Clase para entrenar el modelo con productos personalizados"""
    
    def __init__(self, base_model_path='yolov8n.pt'):
        """
        Inicializa el entrenador personalizado
        
        Args:
            base_model_path (str): Ruta al modelo base YOLO
        """
        self.base_model_path = base_model_path
        self.training_data_dir = 'training_data'
        self.custom_data_dir = 'custom_products'  # Directorio para productos personalizados
        self.custom_model_path = 'custom_products.pt'
        self.is_training = False
        
        # Crear directorios necesarios
        self._create_directories()
        
        # Cargar modelo base
        self.model = YOLO(base_model_path)
        
        # Mapeo de productos personalizados
        self.custom_products = self._load_custom_products()
    
    def _create_directories(self):
        """Crea los directorios necesarios para el entrenamiento"""
        directories = [
            self.training_data_dir,
            os.path.join(self.training_data_dir, 'images'),
            os.path.join(self.training_data_dir, 'labels'),
            os.path.join(self.training_data_dir, 'raw_images'),
            self.custom_data_dir  # Directorio para productos personalizados
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _load_custom_products(self):
        """Carga la lista de productos personalizados"""
        products_file = os.path.join(self.training_data_dir, 'custom_products.json')
        
        if os.path.exists(products_file):
            with open(products_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    
    def _save_custom_products(self):
        """Guarda la lista de productos personalizados"""
        products_file = os.path.join(self.training_data_dir, 'custom_products.json')
        
        with open(products_file, 'w', encoding='utf-8') as f:
            json.dump(self.custom_products, f, ensure_ascii=False, indent=2)
    
    def add_product_for_training(self, product_name, images, labels=None):
        """
        Agrega un producto para entrenamiento
        
        Args:
            product_name (str): Nombre del producto
            images (list): Lista de imágenes (numpy arrays o rutas)
            labels (list): Lista de etiquetas (opcional)
        """
        if product_name in self.custom_products:
            print(f"Producto '{product_name}' ya existe. Actualizando...")
        
        # Crear directorio para el producto
        product_dir = os.path.join(self.training_data_dir, 'raw_images', product_name)
        os.makedirs(product_dir, exist_ok=True)
        
        # Guardar imágenes
        image_paths = []
        for i, image in enumerate(images):
            if isinstance(image, str):
                # Es una ruta de archivo
                image_path = image
            else:
                # Es un numpy array
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{product_name}_{i}_{timestamp}.jpg"
                image_path = os.path.join(product_dir, filename)
                cv2.imwrite(image_path, image)
            
            image_paths.append(image_path)
        
        # Guardar información del producto
        self.custom_products[product_name] = {
            'name': product_name,
            'image_count': len(image_paths),
            'image_paths': image_paths,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        # Guardar cambios
        self._save_custom_products()
        
        print(f"Producto '{product_name}' agregado con {len(image_paths)} imágenes")
        return True
    
    def capture_images_from_web(self, product_name, images, num_images):
        """
        Captura imágenes desde la interfaz web
        
        Args:
            product_name (str): Nombre del producto
            images (list): Lista de imágenes en base64
            num_images (int): Número de imágenes capturadas
        """
        try:
            import base64
            from datetime import datetime
            
            # Crear directorio del producto en custom_products
            product_dir = os.path.join(self.custom_data_dir, product_name)
            os.makedirs(product_dir, exist_ok=True)
            
            image_paths = []
            for i, image_data in enumerate(images):
                # Decodificar imagen base64
                if image_data.startswith('data:image'):
                    # Remover el prefijo data:image/jpeg;base64,
                    image_data = image_data.split(',')[1]
                
                # Decodificar base64
                image_bytes = base64.b64decode(image_data)
                
                # Guardar imagen
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{product_name}_{i}_{timestamp}.jpg"
                image_path = os.path.join(product_dir, filename)
                
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                
                image_paths.append(image_path)
            
            # Guardar información del producto
            self.custom_products[product_name] = {
                'name': product_name,
                'image_count': len(image_paths),
                'image_paths': image_paths,
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
            
            # Guardar cambios
            self._save_custom_products()
            
            print(f"Producto '{product_name}' agregado con {len(image_paths)} imágenes desde web")
            return True
            
        except Exception as e:
            print(f"Error al procesar imágenes desde web: {e}")
            return False
    
    def capture_product_images(self, product_name, camera_index=0, num_images=10):
        """
        Captura imágenes de un producto desde la cámara web
        
        Args:
            product_name (str): Nombre del producto
            camera_index (int): Índice de la cámara
            num_images (int): Número de imágenes a capturar
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise Exception(f"No se pudo abrir la cámara {camera_index}")
        
        try:
            print(f"Capturando {num_images} imágenes para '{product_name}'...")
            print("Presiona ESPACIO para capturar una imagen, ESC para salir")
            
            captured_images = []
            image_count = 0
            
            while image_count < num_images:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mostrar frame con instrucciones
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Producto: {product_name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Imagenes capturadas: {image_count}/{num_images}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "ESPACIO: Capturar | ESC: Salir", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Captura de Producto', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Espacio para capturar
                    captured_images.append(frame.copy())
                    image_count += 1
                    print(f"Imagen {image_count} capturada")
                    
                elif key == 27:  # ESC para salir
                    break
            
            cv2.destroyAllWindows()
            
            if captured_images:
                self.add_product_for_training(product_name, captured_images)
                return True
            else:
                print("No se capturaron imágenes")
                return False
                
        finally:
            cap.release()
    
    def capture_images_from_web(self, product_name, frames_data, num_images=10):
        """
        Captura imágenes desde la interfaz web
        
        Args:
            product_name (str): Nombre del producto
            frames_data (list): Lista de frames en base64 desde la web
            num_images (int): Número de imágenes a procesar
        """
        try:
            captured_images = []
            
            for i, frame_data in enumerate(frames_data[:num_images]):
                # Decodificar imagen base64
                import base64
                image_data = base64.b64decode(frame_data.split(',')[1])
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    captured_images.append(image)
            
            if captured_images:
                self.add_product_for_training(product_name, captured_images)
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error capturando imágenes desde web: {e}")
            return False
    
    def prepare_training_data(self):
        """
        Prepara los datos para el entrenamiento
        
        Returns:
            bool: True si se prepararon los datos correctamente
        """
        if not self.custom_products:
            print("No hay productos personalizados para entrenar")
            return False
        
        print("Preparando datos de entrenamiento...")
        
        # Crear estructura YOLO
        images_dir = os.path.join(self.training_data_dir, 'images')
        labels_dir = os.path.join(self.training_data_dir, 'labels')
        
        # Limpiar directorios
        for directory in [images_dir, labels_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
        
        # Copiar imágenes y crear etiquetas
        for product_id, (product_name, product_info) in enumerate(self.custom_products.items()):
            for image_path in product_info['image_paths']:
                if os.path.exists(image_path):
                    # Copiar imagen
                    image_filename = os.path.basename(image_path)
                    dest_image_path = os.path.join(images_dir, image_filename)
                    shutil.copy2(image_path, dest_image_path)
                    
                    # Crear etiqueta YOLO (asumiendo que el objeto ocupa todo el frame)
                    label_filename = image_filename.replace('.jpg', '.txt')
                    label_path = os.path.join(labels_dir, label_filename)
                    
                    # Formato YOLO: class_id center_x center_y width height
                    # Asumiendo que el objeto ocupa todo el frame
                    with open(label_path, 'w') as f:
                        f.write(f"{product_id} 0.5 0.5 1.0 1.0\n")
        
        print(f"Datos de entrenamiento preparados: {len(self.custom_products)} productos")
        return True
    
    def train_custom_model(self, epochs=50, batch_size=16):
        """
        Entrena el modelo personalizado
        
        Args:
            epochs (int): Número de épocas de entrenamiento
            batch_size (int): Tamaño del batch
            
        Returns:
            bool: True si el entrenamiento fue exitoso
        """
        if self.is_training:
            print("Ya hay un entrenamiento en progreso")
            return False
        
        if not self.prepare_training_data():
            return False
        
        try:
            self.is_training = True
            print("Iniciando entrenamiento del modelo personalizado...")
            
            # Configurar entrenamiento
            data_yaml = os.path.join(self.training_data_dir, 'data.yaml')
            self._create_data_yaml(data_yaml)
            
            # Entrenar modelo
            results = self.model.train(
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
                shutil.copy2(trained_model_path, self.custom_model_path)
                print(f"Modelo personalizado guardado en: {self.custom_model_path}")
                return True
            else:
                print("Error: No se pudo encontrar el modelo entrenado")
                return False
                
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
            return False
        finally:
            self.is_training = False
    
    def _create_data_yaml(self, yaml_path):
        """Crea el archivo data.yaml para el entrenamiento"""
        data_content = f"""
path: {os.path.abspath(self.training_data_dir)}
train: images
val: images

nc: {len(self.custom_products)}
names: {list(self.custom_products.keys())}
"""
        
        with open(yaml_path, 'w') as f:
            f.write(data_content)
    
    def load_custom_model(self):
        """Carga el modelo personalizado si existe"""
        if os.path.exists(self.custom_model_path):
            self.model = YOLO(self.custom_model_path)
            print("Modelo personalizado cargado")
            return True
        else:
            print("No se encontró modelo personalizado, usando modelo base")
            return False
    
    def get_training_status(self):
        """Obtiene el estado del entrenamiento"""
        return {
            'is_training': self.is_training,
            'custom_products_count': len(self.custom_products),
            'custom_model_exists': os.path.exists(self.custom_model_path),
            'products': list(self.custom_products.keys())
        }
    
    def get_detailed_products(self):
        """Obtiene información detallada de los productos personalizados"""
        products = []
        for product_name, product_info in self.custom_products.items():
            products.append({
                'name': product_name,
                'image_count': product_info.get('image_count', 0),
                'created_at': product_info.get('created_at', ''),
                'last_updated': product_info.get('last_updated', '')
            })
        return products
    
    def get_product_images(self, product_name):
        """Obtiene las imágenes de un producto específico"""
        if product_name not in self.custom_products:
            return []
        
        product_info = self.custom_products[product_name]
        image_paths = product_info.get('image_paths', [])
        
        # Obtener solo los nombres de archivo
        image_names = []
        for image_path in image_paths:
            if os.path.exists(image_path):
                image_names.append(os.path.basename(image_path))
            else:
                # Si la imagen no existe en la ruta original, buscar en custom_products
                filename = os.path.basename(image_path)
                custom_path = os.path.join(self.custom_data_dir, product_name, filename)
                if os.path.exists(custom_path):
                    image_names.append(filename)
        
        return image_names
    
    def delete_product(self, product_name):
        """Elimina un producto del entrenamiento"""
        if product_name in self.custom_products:
            # Eliminar directorio del producto
            product_dir = os.path.join(self.training_data_dir, 'raw_images', product_name)
            if os.path.exists(product_dir):
                shutil.rmtree(product_dir)
            
            # Eliminar de la lista
            del self.custom_products[product_name]
            self._save_custom_products()
            
            print(f"Producto '{product_name}' eliminado")
            return True
        else:
            print(f"Producto '{product_name}' no encontrado")
            return False
