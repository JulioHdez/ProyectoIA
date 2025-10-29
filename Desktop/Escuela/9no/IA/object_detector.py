import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import json
from custom_trainer import CustomProductTrainer

class ObjectDetector:
    """Clase para detección de objetos usando YOLO"""
    
    def __init__(self, model_path='yolov8n.pt'):
        """
        Inicializa el detector de objetos
        
        Args:
            model_path (str): Ruta al modelo YOLO
        """
        self.base_model_path = model_path
        self.detection_threshold = 0.5
        
        # Inicializar entrenador personalizado
        self.custom_trainer = CustomProductTrainer(model_path)
        
        # Intentar cargar modelo personalizado primero
        if self.custom_trainer.load_custom_model():
            self.model = self.custom_trainer.model
            self.class_names = self.model.names
            print("Usando modelo personalizado")
        else:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            print("Usando modelo base")
        
        # Mapeo de clases COCO a productos comunes
        self.product_mapping = {
            'apple': 'manzana',
            'banana': 'plátano',
            'orange': 'naranja',
            'bottle': 'botella',
            'cup': 'vaso',
            'bowl': 'tazón',
            'book': 'libro',
            'laptop': 'laptop',
            'mouse': 'ratón',
            'keyboard': 'teclado',
            'cell phone': 'teléfono',
            'backpack': 'mochila',
            'handbag': 'bolso',
            'tie': 'corbata',
            'suitcase': 'maleta',
            'frisbee': 'frisbee',
            'sports ball': 'pelota',
            'tennis racket': 'raqueta',
            'wine glass': 'copa',
            'fork': 'tenedor',
            'knife': 'cuchillo',
            'spoon': 'cuchara',
            'chair': 'silla',
            'couch': 'sofá',
            'bed': 'cama',
            'dining table': 'mesa',
            'toilet': 'inodoro',
            'tv': 'televisión',
            'remote': 'control remoto',
            'book': 'libro',
            'clock': 'reloj',
            'vase': 'jarrón',
            'scissors': 'tijeras',
            'teddy bear': 'oso de peluche',
            'hair drier': 'secador',
            'toothbrush': 'cepillo de dientes'
        }
    
    def reload_model(self):
        """Recarga el modelo personalizado si existe"""
        if self.custom_trainer.load_custom_model():
            self.model = self.custom_trainer.model
            self.class_names = self.model.names
            print("Modelo personalizado recargado")
            return True
        else:
            print("No se encontró modelo personalizado")
            return False
    
    def detect_objects(self, image):
        """
        Detecta objetos en una imagen
        
        Args:
            image: Imagen de entrada (numpy array o ruta)
            
        Returns:
            list: Lista de detecciones con información de objetos
        """
        try:
            # Ejecutar detección
            results = self.model(image, conf=self.detection_threshold)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Obtener coordenadas del bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Obtener confianza y clase
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names[class_id]
                        
                        # Si es un modelo personalizado, usar el nombre de la clase directamente
                        if hasattr(self.custom_trainer, 'model') and self.custom_trainer.model == self.model:
                            product_name = class_name
                        else:
                            # Mapear a nombre de producto en español para modelo base
                            product_name = self.product_mapping.get(class_name, class_name)
                        
                        detection = {
                            'class_id': class_id,
                            'class_name': class_name,
                            'product_name': product_name,
                            'confidence': float(confidence),
                            'bbox': {
                                'x1': int(x1),
                                'y1': int(y1),
                                'x2': int(x2),
                                'y2': int(y2)
                            },
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error en detección: {e}")
            return []
    
    def detect_from_camera(self, camera_index=0):
        """
        Detecta objetos desde la cámara web
        
        Args:
            camera_index (int): Índice de la cámara
            
        Yields:
            tuple: (frame, detections) para cada frame
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise Exception(f"No se pudo abrir la cámara {camera_index}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detectar objetos en el frame
                detections = self.detect_objects(frame)
                
                # Dibujar bounding boxes
                annotated_frame = self.draw_detections(frame, detections)
                
                yield annotated_frame, detections
                
        finally:
            cap.release()
    
    def draw_detections(self, image, detections):
        """
        Dibuja las detecciones en la imagen
        
        Args:
            image: Imagen original
            detections: Lista de detecciones
            
        Returns:
            numpy array: Imagen con detecciones dibujadas
        """
        annotated_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            # Color basado en la confianza
            confidence = detection['confidence']
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
            
            # Dibujar bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar etiqueta
            label = f"{detection['product_name']}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Fondo para el texto
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Texto
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_image
    
    def save_detection_image(self, image, detections, filename=None):
        """
        Guarda la imagen con detecciones
        
        Args:
            image: Imagen con detecciones
            detections: Lista de detecciones
            filename: Nombre del archivo (opcional)
            
        Returns:
            str: Ruta del archivo guardado
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}.jpg"
        
        # Crear directorio si no existe
        os.makedirs('uploads', exist_ok=True)
        
        filepath = os.path.join('uploads', filename)
        cv2.imwrite(filepath, image)
        
        # Guardar metadatos de detección
        metadata_path = filepath.replace('.jpg', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(detections, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def get_product_suggestions(self, detections):
        """
        Obtiene sugerencias de productos basadas en las detecciones
        
        Args:
            detections: Lista de detecciones
            
        Returns:
            list: Lista de productos sugeridos
        """
        suggestions = []
        
        for detection in detections:
            if detection['confidence'] > self.detection_threshold:
                suggestion = {
                    'name': detection['product_name'],
                    'confidence': detection['confidence'],
                    'detected_class': detection['class_name'],
                    'suggested_price': self._get_suggested_price(detection['product_name'])
                }
                suggestions.append(suggestion)
        
        return suggestions
    
    def _get_suggested_price(self, product_name):
        """
        Obtiene un precio sugerido basado en el nombre del producto
        
        Args:
            product_name (str): Nombre del producto
            
        Returns:
            float: Precio sugerido
        """
        # Precios sugeridos por categoría
        price_suggestions = {
            'manzana': 2.50,
            'plátano': 1.80,
            'naranja': 2.00,
            'botella': 5.00,
            'vaso': 3.50,
            'tazón': 8.00,
            'libro': 15.00,
            'laptop': 800.00,
            'ratón': 25.00,
            'teclado': 45.00,
            'teléfono': 300.00,
            'mochila': 35.00,
            'bolso': 50.00,
            'corbata': 20.00,
            'maleta': 80.00,
            'frisbee': 12.00,
            'pelota': 15.00,
            'raqueta': 60.00,
            'copa': 8.00,
            'tenedor': 2.00,
            'cuchillo': 5.00,
            'cuchara': 2.00,
            'silla': 45.00,
            'sofá': 200.00,
            'cama': 300.00,
            'mesa': 120.00,
            'inodoro': 150.00,
            'televisión': 400.00,
            'control remoto': 15.00,
            'reloj': 25.00,
            'jarrón': 30.00,
            'tijeras': 8.00,
            'oso de peluche': 20.00,
            'secador': 40.00,
            'cepillo de dientes': 5.00
        }
        
        return price_suggestions.get(product_name.lower(), 10.00)  # Precio por defecto
    
    def add_custom_product(self, product_name, images):
        """
        Agrega un producto personalizado para entrenamiento
        
        Args:
            product_name (str): Nombre del producto
            images (list): Lista de imágenes del producto
            
        Returns:
            bool: True si se agregó correctamente
        """
        return self.custom_trainer.add_product_for_training(product_name, images)
    
    def capture_product_images(self, product_name, camera_index=0, num_images=10):
        """
        Captura imágenes de un producto desde la cámara
        
        Args:
            product_name (str): Nombre del producto
            camera_index (int): Índice de la cámara
            num_images (int): Número de imágenes a capturar
            
        Returns:
            bool: True si se capturaron las imágenes correctamente
        """
        return self.custom_trainer.capture_product_images(product_name, camera_index, num_images)
    
    def train_custom_model(self, epochs=50, batch_size=16):
        """
        Entrena el modelo con productos personalizados
        
        Args:
            epochs (int): Número de épocas
            batch_size (int): Tamaño del batch
            
        Returns:
            bool: True si el entrenamiento fue exitoso
        """
        success = self.custom_trainer.train_custom_model(epochs, batch_size)
        if success:
            # Recargar el modelo personalizado
            self.custom_trainer.load_custom_model()
        return success
    
    def get_training_status(self):
        """Obtiene el estado del entrenamiento"""
        return self.custom_trainer.get_training_status()
    
    def delete_custom_product(self, product_name):
        """Elimina un producto personalizado"""
        return self.custom_trainer.delete_product(product_name)

class CameraHandler:
    """Manejador para la cámara web"""
    
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
    
    def start_camera(self):
        """Inicia la cámara"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise Exception(f"No se pudo abrir la cámara {self.camera_index}")
        self.is_running = True
    
    def stop_camera(self):
        """Detiene la cámara"""
        if self.cap:
            self.cap.release()
        self.is_running = False
    
    def get_frame(self):
        """Obtiene un frame de la cámara"""
        if not self.cap or not self.is_running:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def is_camera_available(self):
        """Verifica si la cámara está disponible"""
        cap = cv2.VideoCapture(self.camera_index)
        available = cap.isOpened()
        cap.release()
        return available

