import cv2
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configurar matplotlib para usar backend sin GUI (evita problemas con tkinter en threads)
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt

class TensorFlowTrainer:
    """Clase para entrenar modelos de clasificación con TensorFlow/Keras"""
    
    def __init__(self, base_data_dir='custom_products', model_type='mobilenet'):
        """
        Inicializa el entrenador de TensorFlow
        
        Args:
            base_data_dir (str): Directorio base donde están los productos
            model_type (str): Tipo de modelo base ('mobilenet', 'efficientnet', 'resnet', 'custom')
        """
        self.base_data_dir = base_data_dir
        self.model_type = model_type
        self.models_dir = 'tensorflow_models'
        self.training_history_dir = 'tensorflow_training_history'
        self.is_training = False
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
        # Crear directorios necesarios
        self._create_directories()
        
        # Configuración de entrenamiento
        self.img_size = (224, 224)  # Tamaño estándar para modelos pre-entrenados
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
        
    def _create_directories(self):
        """Crea los directorios necesarios para TensorFlow"""
        directories = [
            self.models_dir,
            self.training_history_dir,
            os.path.join(self.models_dir, 'saved_models'),
            os.path.join(self.models_dir, 'checkpoints')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_product_images(self, product_dirs=None):
        """
        Carga imágenes de productos desde las carpetas
        
        Args:
            product_dirs (list): Lista de nombres de productos a cargar. Si None, carga todos.
            
        Returns:
            tuple: (images, labels) donde images es array de numpy y labels son los nombres de productos
        """
        images = []
        labels = []
        
        if not os.path.exists(self.base_data_dir):
            print(f"Directorio {self.base_data_dir} no existe")
            return np.array([]), np.array([])
        
        # Obtener lista de productos
        if product_dirs is None:
            product_dirs = [d for d in os.listdir(self.base_data_dir) 
                          if os.path.isdir(os.path.join(self.base_data_dir, d))]
        
        print(f"Cargando imágenes de {len(product_dirs)} productos...")
        
        for product_name in product_dirs:
            product_path = os.path.join(self.base_data_dir, product_name)
            
            # Buscar imágenes en diferentes ubicaciones posibles
            image_paths = []
            
            # Buscar en images/
            images_dir = os.path.join(product_path, 'images')
            if os.path.exists(images_dir):
                image_paths.extend([
                    os.path.join(images_dir, f) 
                    for f in os.listdir(images_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                ])
            
            # Buscar en training/images/
            training_images_dir = os.path.join(product_path, 'training', 'images')
            if os.path.exists(training_images_dir):
                image_paths.extend([
                    os.path.join(training_images_dir, f) 
                    for f in os.listdir(training_images_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                ])
            
            # Si no hay imágenes en subcarpetas, buscar directamente en la carpeta del producto
            if not image_paths:
                image_paths = [
                    os.path.join(product_path, f) 
                    for f in os.listdir(product_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                ]
            
            print(f"  {product_name}: {len(image_paths)} imágenes encontradas")
            
            # Cargar imágenes
            for img_path in image_paths:
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Convertir BGR a RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Redimensionar
                        img = cv2.resize(img, self.img_size)
                        images.append(img)
                        labels.append(product_name)
                except Exception as e:
                    print(f"    Error cargando {img_path}: {e}")
                    continue
        
        if len(images) == 0:
            print("No se encontraron imágenes")
            return np.array([]), np.array([])
        
        images = np.array(images, dtype=np.float32) / 255.0  # Normalizar a [0, 1]
        labels = np.array(labels)
        
        print(f"Total de imágenes cargadas: {len(images)}")
        print(f"Productos únicos: {len(np.unique(labels))}")
        
        return images, labels
    
    def prepare_data(self, images, labels, test_size=0.2, val_size=0.1):
        """
        Prepara los datos para entrenamiento
        
        Args:
            images: Array de imágenes
            labels: Array de etiquetas
            test_size: Proporción para test
            val_size: Proporción para validación (del conjunto de entrenamiento)
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test, class_names)
        """
        # Codificar etiquetas
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        self.class_names = self.label_encoder.classes_.tolist()
        
        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            images, encoded_labels, test_size=test_size, random_state=42, stratify=encoded_labels
        )
        
        # Dividir train en train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )
        
        # Convertir a one-hot encoding
        num_classes = len(self.class_names)
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_val = keras.utils.to_categorical(y_val, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        
        print(f"\nDivisión de datos:")
        print(f"  Entrenamiento: {len(X_train)} imágenes")
        print(f"  Validación: {len(X_val)} imágenes")
        print(f"  Prueba: {len(X_test)} imágenes")
        print(f"  Clases: {num_classes} ({', '.join(self.class_names)})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, self.class_names
    
    def build_model(self, num_classes, use_pretrained=True):
        """
        Construye el modelo de clasificación
        
        Args:
            num_classes: Número de clases a clasificar
            use_pretrained: Si usar modelo pre-entrenado como base
            
        Returns:
            Modelo de Keras
        """
        input_shape = (*self.img_size, 3)
        
        if use_pretrained:
            if self.model_type == 'mobilenet':
                base_model = MobileNetV2(
                    input_shape=input_shape,
                    include_top=False,
                    weights='imagenet'
                )
            elif self.model_type == 'efficientnet':
                base_model = EfficientNetB0(
                    input_shape=input_shape,
                    include_top=False,
                    weights='imagenet'
                )
            elif self.model_type == 'resnet':
                base_model = ResNet50(
                    input_shape=input_shape,
                    include_top=False,
                    weights='imagenet'
                )
            else:
                # Modelo personalizado simple
                return self._build_custom_model(num_classes, input_shape)
            
            # Congelar capas base (opcional, se puede descongelar después)
            base_model.trainable = False
            
            # Construir modelo completo
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.2),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(num_classes, activation='softmax')
            ])
        else:
            model = self._build_custom_model(num_classes, input_shape)
        
        # Compilar modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def _build_custom_model(self, num_classes, input_shape):
        """Construye un modelo personalizado sin pre-entrenamiento"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def train(self, epochs=None, batch_size=None, product_dirs=None, use_augmentation=True):
        """
        Entrena el modelo
        
        Args:
            epochs: Número de épocas (usa self.epochs si None)
            batch_size: Tamaño de batch (usa self.batch_size si None)
            product_dirs: Lista de productos a entrenar (None = todos)
            use_augmentation: Si usar data augmentation
            
        Returns:
            dict: Historial de entrenamiento
        """
        if self.is_training:
            print("Ya hay un entrenamiento en progreso")
            return None
        
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        
        try:
            self.is_training = True
            print("=" * 60)
            print("INICIANDO ENTRENAMIENTO CON TENSORFLOW")
            print("=" * 60)
            
            # Cargar datos
            images, labels = self.load_product_images(product_dirs)
            if len(images) == 0:
                print("Error: No se encontraron imágenes para entrenar")
                return None
            
            # Preparar datos
            X_train, X_val, X_test, y_train, y_val, y_test, class_names = self.prepare_data(images, labels)
            
            # Construir modelo
            print(f"\nConstruyendo modelo ({self.model_type})...")
            self.model = self.build_model(len(class_names), use_pretrained=True)
            self.model.summary()
            
            # Data augmentation
            if use_augmentation:
                train_datagen = ImageDataGenerator(
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest'
                )
            else:
                train_datagen = ImageDataGenerator()
            
            # Callbacks
            checkpoint_path = os.path.join(self.models_dir, 'checkpoints', 'best_model.h5')
            callbacks_list = [
                callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Ajustar batch_size si es mayor que el número de imágenes
            if batch_size > len(X_train):
                print(f"  Advertencia: batch_size ({batch_size}) es mayor que imágenes de entrenamiento ({len(X_train)})")
                batch_size = max(1, len(X_train) // 2)  # Usar la mitad de las imágenes como batch_size mínimo
                print(f"  Ajustando batch_size a: {batch_size}")
            
            # Calcular steps_per_epoch (mínimo 1)
            steps_per_epoch = max(1, len(X_train) // batch_size)
            if steps_per_epoch == 0:
                steps_per_epoch = 1
            
            # Entrenar
            print(f"\nIniciando entrenamiento...")
            print(f"  Épocas: {epochs}")
            print(f"  Batch size: {batch_size}")
            print(f"  Steps per epoch: {steps_per_epoch}")
            print(f"  Data augmentation: {use_augmentation}")
            
            # Si hay muy pocas imágenes, usar fit directo sin ImageDataGenerator.flow
            if len(X_train) < batch_size * 2:
                print("  Usando modo de entrenamiento directo (pocas imágenes)")
                history = self.model.fit(
                    X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks_list,
                    verbose=1
                )
            else:
                history = self.model.fit(
                    train_datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks_list,
                    verbose=1
                )
            
            # Evaluar en test
            print("\nEvaluando en conjunto de prueba...")
            test_loss, test_accuracy, test_top_k = self.model.evaluate(X_test, y_test, verbose=1)
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Test Top-K Accuracy: {test_top_k:.4f}")
            
            # Guardar modelo final
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.models_dir, 'saved_models', f'model_{timestamp}.h5')
            self.model.save(model_path)
            print(f"\nModelo guardado en: {model_path}")
            
            # Guardar también como modelo más reciente
            latest_model_path = os.path.join(self.models_dir, 'saved_models', 'latest_model.h5')
            self.model.save(latest_model_path)
            
            # Guardar metadata
            metadata = {
                'model_type': self.model_type,
                'num_classes': len(class_names),
                'class_names': class_names,
                'img_size': self.img_size,
                'epochs': epochs,
                'batch_size': batch_size,
                'test_accuracy': float(test_accuracy),
                'test_top_k_accuracy': float(test_top_k),
                'timestamp': timestamp,
                'model_path': model_path
            }
            
            metadata_path = os.path.join(self.models_dir, 'saved_models', f'metadata_{timestamp}.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Guardar historial
            self._save_training_history(history, timestamp)
            
            print("=" * 60)
            print("ENTRENAMIENTO COMPLETADO")
            print("=" * 60)
            
            return {
                'history': history.history,
                'test_accuracy': float(test_accuracy),
                'test_top_k_accuracy': float(test_top_k),
                'model_path': model_path,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.is_training = False
    
    def _save_training_history(self, history, timestamp):
        """Guarda el historial de entrenamiento"""
        # Guardar como JSON
        history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
        history_path = os.path.join(self.training_history_dir, f'history_{timestamp}.json')
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        # Guardar gráficos
        self._plot_training_history(history, timestamp)
    
    def _plot_training_history(self, history, timestamp):
        """Genera y guarda gráficos del historial de entrenamiento"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.training_history_dir, f'training_history_{timestamp}.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Gráficos guardados en: {plot_path}")
    
    def load_model(self, model_path=None):
        """
        Carga un modelo entrenado
        
        Args:
            model_path: Ruta al modelo. Si None, carga el más reciente.
            
        Returns:
            bool: True si se cargó correctamente
        """
        if model_path is None:
            latest_path = os.path.join(self.models_dir, 'saved_models', 'latest_model.h5')
            if os.path.exists(latest_path):
                model_path = latest_path
            else:
                print("No se encontró modelo reciente")
                return False
        
        if not os.path.exists(model_path):
            print(f"Modelo no encontrado: {model_path}")
            return False
        
        try:
            self.model = keras.models.load_model(model_path)
            
            # Cargar metadata si existe
            metadata_path = model_path.replace('.h5', '_metadata.json')
            if not os.path.exists(metadata_path):
                # Buscar metadata con timestamp
                base_name = os.path.basename(model_path).replace('.h5', '')
                metadata_dir = os.path.dirname(model_path)
                metadata_files = [f for f in os.listdir(metadata_dir) 
                                if f.startswith('metadata_') and f.endswith('.json')]
                if metadata_files:
                    metadata_path = os.path.join(metadata_dir, metadata_files[-1])
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.class_names = metadata.get('class_names', [])
                    self.model_type = metadata.get('model_type', 'unknown')
            
            print(f"Modelo cargado desde: {model_path}")
            if self.class_names:
                print(f"Clases: {', '.join(self.class_names)}")
            return True
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
    
    def predict(self, image):
        """
        Predice la clase de una imagen
        
        Args:
            image: Imagen (numpy array o ruta)
            
        Returns:
            dict: Predicciones con confianza
        """
        if self.model is None:
            print("Error: No hay modelo cargado")
            return None
        
        # Cargar imagen si es ruta
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                print(f"Error: No se pudo cargar imagen {image}")
                return None
        else:
            img = image.copy()
        
        # Preprocesar
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predecir
        predictions = self.model.predict(img, verbose=0)[0]
        
        # Obtener top 5 predicciones
        top_indices = np.argsort(predictions)[::-1][:5]
        
        results = []
        for idx in top_indices:
            if idx < len(self.class_names):
                results.append({
                    'class_name': self.class_names[idx],
                    'confidence': float(predictions[idx])
                })
        
        return results
    
    def get_training_status(self):
        """Obtiene el estado del entrenamiento"""
        return {
            'is_training': self.is_training,
            'model_loaded': self.model is not None,
            'model_type': self.model_type,
            'num_classes': len(self.class_names) if self.class_names else 0,
            'class_names': self.class_names
        }

