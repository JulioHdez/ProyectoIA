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
            self.custom_data_dir  # Directorio base para productos personalizados
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _create_product_directories(self, product_name):
        """
        Crea la estructura de carpetas para un producto específico
        
        Estructura:
        custom_products/
          Producto/
            ├── images/          # Imágenes originales capturadas
            ├── labels/           # Etiquetas YOLO
            ├── training/        # Datos preparados para entrenamiento
            │   ├── images/
            │   └── labels/
            ├── models/          # Modelos entrenados específicos del producto
            └── metadata.json    # Información del producto
        
        Args:
            product_name (str): Nombre del producto
        """
        # Normalizar nombre del producto para usar como nombre de carpeta
        safe_name = self._sanitize_product_name(product_name)
        
        product_base_dir = os.path.join(self.custom_data_dir, safe_name)
        
        # Estructura de subcarpetas para cada producto
        subdirectories = [
            product_base_dir,
            os.path.join(product_base_dir, 'images'),      # Imágenes originales
            os.path.join(product_base_dir, 'labels'),       # Etiquetas YOLO
            os.path.join(product_base_dir, 'training'),     # Datos de entrenamiento
            os.path.join(product_base_dir, 'training', 'images'),
            os.path.join(product_base_dir, 'training', 'labels'),
            os.path.join(product_base_dir, 'models'),       # Modelos entrenados
        ]
        
        for directory in subdirectories:
            os.makedirs(directory, exist_ok=True)
        
        return product_base_dir
    
    def _sanitize_product_name(self, product_name):
        """
        Sanitiza el nombre del producto para usarlo como nombre de carpeta
        
        Args:
            product_name (str): Nombre original del producto
            
        Returns:
            str: Nombre sanitizado
        """
        # Reemplazar caracteres especiales y espacios
        safe_name = product_name.replace(' ', '_')
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c in ('_', '-'))
        return safe_name
    
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
        Agrega un producto para entrenamiento con estructura de carpetas por producto
        
        Args:
            product_name (str): Nombre del producto
            images (list): Lista de imágenes (numpy arrays o rutas)
            labels (list): Lista de etiquetas (opcional)
        """
        if product_name in self.custom_products:
            print(f"Producto '{product_name}' ya existe. Actualizando...")
        
        # Crear estructura de carpetas para este producto específico
        product_base_dir = self._create_product_directories(product_name)
        images_dir = os.path.join(product_base_dir, 'images')
        labels_dir = os.path.join(product_base_dir, 'labels')
        
        # Guardar imágenes en la carpeta del producto
        image_paths = []
        for i, image in enumerate(images):
            if isinstance(image, str):
                # Es una ruta de archivo - copiar a la carpeta del producto
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self._sanitize_product_name(product_name)}_{i}_{timestamp}.jpg"
                dest_path = os.path.join(images_dir, filename)
                shutil.copy2(image, dest_path)
                image_path = dest_path
            else:
                # Es un numpy array - guardar directamente
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self._sanitize_product_name(product_name)}_{i}_{timestamp}.jpg"
                image_path = os.path.join(images_dir, filename)
                cv2.imwrite(image_path, image)
            
            image_paths.append(image_path)
            
            # Si hay etiquetas, guardarlas también
            if labels and i < len(labels):
                label_filename = filename.replace('.jpg', '.txt')
                label_path = os.path.join(labels_dir, label_filename)
                with open(label_path, 'w') as f:
                    f.write(labels[i])
        
        # Guardar metadata del producto en su propia carpeta
        metadata_path = os.path.join(product_base_dir, 'metadata.json')
        product_metadata = {
            'name': product_name,
            'safe_name': self._sanitize_product_name(product_name),
            'image_count': len(image_paths),
            'image_paths': image_paths,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'base_dir': product_base_dir
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(product_metadata, f, ensure_ascii=False, indent=2)
        
        # Actualizar lista global de productos
        self.custom_products[product_name] = {
            'name': product_name,
            'safe_name': self._sanitize_product_name(product_name),
            'image_count': len(image_paths),
            'image_paths': image_paths,
            'base_dir': product_base_dir,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        # Guardar cambios
        self._save_custom_products()
        
        print(f"Producto '{product_name}' agregado con {len(image_paths)} imágenes")
        print(f"Carpeta del producto: {product_base_dir}")
        return True
    
    def capture_images_from_web(self, product_name, images, num_images):
        """
        Captura imágenes desde la interfaz web usando estructura de carpetas por producto
        
        Args:
            product_name (str): Nombre del producto
            images (list): Lista de imágenes en base64
            num_images (int): Número de imágenes capturadas
        """
        try:
            import base64
            from datetime import datetime
            
            # Crear estructura de carpetas para este producto específico
            product_base_dir = self._create_product_directories(product_name)
            images_dir = os.path.join(product_base_dir, 'images')
            
            image_paths = []
            for i, image_data in enumerate(images):
                # Decodificar imagen base64
                if image_data.startswith('data:image'):
                    # Remover el prefijo data:image/jpeg;base64,
                    image_data = image_data.split(',')[1]
                
                # Decodificar base64
                image_bytes = base64.b64decode(image_data)
                
                # Guardar imagen en la carpeta del producto
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self._sanitize_product_name(product_name)}_{i}_{timestamp}.jpg"
                image_path = os.path.join(images_dir, filename)
                
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                
                image_paths.append(image_path)
            
            # Guardar metadata del producto en su propia carpeta
            metadata_path = os.path.join(product_base_dir, 'metadata.json')
            product_metadata = {
                'name': product_name,
                'safe_name': self._sanitize_product_name(product_name),
                'image_count': len(image_paths),
                'image_paths': image_paths,
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'base_dir': product_base_dir,
                'source': 'web_capture'
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(product_metadata, f, ensure_ascii=False, indent=2)
            
            # Actualizar lista global de productos
            self.custom_products[product_name] = {
                'name': product_name,
                'safe_name': self._sanitize_product_name(product_name),
                'image_count': len(image_paths),
                'image_paths': image_paths,
                'base_dir': product_base_dir,
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
            
            # Guardar cambios
            self._save_custom_products()
            
            print(f"Producto '{product_name}' agregado con {len(image_paths)} imágenes desde web")
            print(f"Carpeta del producto: {product_base_dir}")
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
    
    def prepare_training_data(self, product_name=None):
        """
        Prepara los datos para el entrenamiento
        
        Args:
            product_name (str, optional): Si se especifica, prepara solo ese producto.
                                        Si es None, prepara todos los productos.
        
        Returns:
            bool: True si se prepararon los datos correctamente
        """
        if not self.custom_products:
            print("No hay productos personalizados para entrenar")
            return False
        
        # Si se especifica un producto, preparar solo ese
        if product_name:
            if product_name not in self.custom_products:
                print(f"Producto '{product_name}' no encontrado")
                return False
            products_to_prepare = {product_name: self.custom_products[product_name]}
        else:
            products_to_prepare = self.custom_products
        
        print(f"Preparando datos de entrenamiento para {len(products_to_prepare)} producto(s)...")
        
        # Crear estructura YOLO global (para entrenamiento conjunto)
        images_dir = os.path.join(self.training_data_dir, 'images')
        labels_dir = os.path.join(self.training_data_dir, 'labels')
        
        # Limpiar directorios globales solo si se preparan todos los productos
        if not product_name:
            for directory in [images_dir, labels_dir]:
                if os.path.exists(directory):
                    shutil.rmtree(directory)
                os.makedirs(directory)
        
        # Preparar datos de cada producto
        for product_id, (prod_name, product_info) in enumerate(products_to_prepare.items()):
            # Obtener carpeta del producto
            product_base_dir = product_info.get('base_dir', 
                os.path.join(self.custom_data_dir, self._sanitize_product_name(prod_name)))
            
            product_images_dir = os.path.join(product_base_dir, 'images')
            product_labels_dir = os.path.join(product_base_dir, 'labels')
            product_training_images_dir = os.path.join(product_base_dir, 'training', 'images')
            product_training_labels_dir = os.path.join(product_base_dir, 'training', 'labels')
            
            # Crear carpetas de entrenamiento del producto
            os.makedirs(product_training_images_dir, exist_ok=True)
            os.makedirs(product_training_labels_dir, exist_ok=True)
            
            # Copiar imágenes y crear etiquetas para este producto
            for image_path in product_info['image_paths']:
                if os.path.exists(image_path):
                    image_filename = os.path.basename(image_path)
                    
                    # Copiar a carpeta de entrenamiento del producto
                    dest_image_path = os.path.join(product_training_images_dir, image_filename)
                    shutil.copy2(image_path, dest_image_path)
                    
                    # Copiar también a carpeta global (para entrenamiento conjunto)
                    if not product_name:
                        global_dest_image_path = os.path.join(images_dir, image_filename)
                        shutil.copy2(image_path, global_dest_image_path)
                    
                    # Crear etiqueta YOLO
                    label_filename = image_filename.replace('.jpg', '.txt')
                    
                    # Etiqueta en carpeta del producto
                    product_label_path = os.path.join(product_training_labels_dir, label_filename)
                    
                    # Verificar si ya existe una etiqueta en la carpeta labels del producto
                    existing_label_path = os.path.join(product_labels_dir, label_filename)
                    if os.path.exists(existing_label_path):
                        # Usar la etiqueta existente
                        shutil.copy2(existing_label_path, product_label_path)
                    else:
                        # Crear etiqueta por defecto (objeto ocupa todo el frame)
                        with open(product_label_path, 'w') as f:
                            f.write(f"{product_id} 0.5 0.5 1.0 1.0\n")
                    
                    # Copiar etiqueta a carpeta global
                    if not product_name:
                        global_label_path = os.path.join(labels_dir, label_filename)
                        shutil.copy2(product_label_path, global_label_path)
            
            print(f"  ✓ Producto '{prod_name}': {len(product_info['image_paths'])} imágenes preparadas")
        
        print(f"Datos de entrenamiento preparados: {len(products_to_prepare)} producto(s)")
        return True
    
    def train_custom_model(self, epochs=50, batch_size=16, product_name=None):
        """
        Entrena el modelo personalizado
        
        Args:
            epochs (int): Número de épocas de entrenamiento
            batch_size (int): Tamaño del batch
            product_name (str, optional): Si se especifica, entrena solo ese producto.
                                         Si es None, entrena todos los productos juntos.
            
        Returns:
            bool: True si el entrenamiento fue exitoso
        """
        if self.is_training:
            print("Ya hay un entrenamiento en progreso")
            return False
        
        # Preparar datos (específico del producto o todos)
        if not self.prepare_training_data(product_name):
            return False
        
        try:
            self.is_training = True
            
            if product_name:
                print(f"Iniciando entrenamiento del modelo para '{product_name}'...")
                product_info = self.custom_products[product_name]
                product_base_dir = product_info.get('base_dir', 
                    os.path.join(self.custom_data_dir, self._sanitize_product_name(product_name)))
                models_dir = os.path.join(product_base_dir, 'models')
                os.makedirs(models_dir, exist_ok=True)
                
                # Crear data.yaml específico del producto
                product_data_yaml = os.path.join(product_base_dir, 'data.yaml')
                self._create_data_yaml(product_data_yaml, [product_name])
                
                # Entrenar modelo específico del producto
                project_name = f"product_{self._sanitize_product_name(product_name)}"
                results = self.model.train(
                    data=product_data_yaml,
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=640,
                    device='cpu',
                    project='custom_training',
                    name=project_name
                )
                
                # Guardar modelo en la carpeta del producto
                trained_model_path = os.path.join('custom_training', project_name, 'weights', 'best.pt')
                if os.path.exists(trained_model_path):
                    product_model_path = os.path.join(models_dir, f'{self._sanitize_product_name(product_name)}.pt')
                    shutil.copy2(trained_model_path, product_model_path)
                    print(f"Modelo de '{product_name}' guardado en: {product_model_path}")
                    return True
                else:
                    print("Error: No se pudo encontrar el modelo entrenado")
                    return False
            else:
                print("Iniciando entrenamiento del modelo conjunto (todos los productos)...")
                
                # Configurar entrenamiento conjunto
                data_yaml = os.path.join(self.training_data_dir, 'data.yaml')
                self._create_data_yaml(data_yaml)
                
                # Entrenar modelo conjunto
                results = self.model.train(
                    data=data_yaml,
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=640,
                    device='cpu',
                    project='custom_training',
                    name='product_detection'
                )
                
                # Guardar modelo entrenado global
                trained_model_path = os.path.join('custom_training', 'product_detection', 'weights', 'best.pt')
                if os.path.exists(trained_model_path):
                    shutil.copy2(trained_model_path, self.custom_model_path)
                    print(f"Modelo conjunto guardado en: {self.custom_model_path}")
                    
                    # También guardar una copia en cada carpeta de producto
                    for prod_name, prod_info in self.custom_products.items():
                        product_base_dir = prod_info.get('base_dir', 
                            os.path.join(self.custom_data_dir, self._sanitize_product_name(prod_name)))
                        models_dir = os.path.join(product_base_dir, 'models')
                        os.makedirs(models_dir, exist_ok=True)
                        product_model_path = os.path.join(models_dir, 'ensemble_model.pt')
                        shutil.copy2(trained_model_path, product_model_path)
                    
                    return True
                else:
                    print("Error: No se pudo encontrar el modelo entrenado")
                    return False
                
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
            return False
        finally:
            self.is_training = False
    
    def _create_data_yaml(self, yaml_path, product_names=None):
        """
        Crea el archivo data.yaml para el entrenamiento
        
        Args:
            yaml_path (str): Ruta donde guardar el archivo YAML
            product_names (list, optional): Lista de nombres de productos.
                                          Si es None, usa todos los productos.
        """
        if product_names is None:
            product_names = list(self.custom_products.keys())
        
        # Determinar el path base según si es para un producto específico o conjunto
        if len(product_names) == 1:
            # Producto individual - usar su carpeta de entrenamiento
            product_name = product_names[0]
            product_info = self.custom_products[product_name]
            product_base_dir = product_info.get('base_dir', 
                os.path.join(self.custom_data_dir, self._sanitize_product_name(product_name)))
            training_dir = os.path.join(product_base_dir, 'training')
            base_path = os.path.abspath(training_dir)
        else:
            # Múltiples productos - usar carpeta global
            base_path = os.path.abspath(self.training_data_dir)
        
        data_content = f"""path: {base_path}
train: images
val: images

nc: {len(product_names)}
names: {product_names}
"""
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
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
        """
        Obtiene las imágenes de un producto específico desde su carpeta
        
        Args:
            product_name (str): Nombre del producto
            
        Returns:
            list: Lista de nombres de archivo de imágenes
        """
        if product_name not in self.custom_products:
            return []
        
        product_info = self.custom_products[product_name]
        
        # Obtener carpeta base del producto
        product_base_dir = product_info.get('base_dir', 
            os.path.join(self.custom_data_dir, self._sanitize_product_name(product_name)))
        images_dir = os.path.join(product_base_dir, 'images')
        
        # Obtener imágenes de la carpeta del producto
        image_names = []
        if os.path.exists(images_dir):
            for filename in os.listdir(images_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_names.append(filename)
        
        # Si no hay imágenes en la carpeta, intentar desde image_paths
        if not image_names:
            image_paths = product_info.get('image_paths', [])
            for image_path in image_paths:
                if os.path.exists(image_path):
                    image_names.append(os.path.basename(image_path))
        
        return sorted(image_names)
    
    def delete_product(self, product_name):
        """
        Elimina un producto del entrenamiento y toda su carpeta
        
        Args:
            product_name (str): Nombre del producto a eliminar
            
        Returns:
            bool: True si se eliminó correctamente
        """
        if product_name in self.custom_products:
            product_info = self.custom_products[product_name]
            
            # Obtener carpeta base del producto
            product_base_dir = product_info.get('base_dir', 
                os.path.join(self.custom_data_dir, self._sanitize_product_name(product_name)))
            
            # Eliminar toda la carpeta del producto (incluye imágenes, labels, training, models, metadata)
            if os.path.exists(product_base_dir):
                shutil.rmtree(product_base_dir)
                print(f"Carpeta del producto eliminada: {product_base_dir}")
            
            # Eliminar de la lista global
            del self.custom_products[product_name]
            self._save_custom_products()
            
            print(f"Producto '{product_name}' eliminado completamente")
            return True
        else:
            print(f"Producto '{product_name}' no encontrado")
            return False
