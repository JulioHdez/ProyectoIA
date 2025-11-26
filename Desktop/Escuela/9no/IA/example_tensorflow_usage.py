"""
Ejemplo de uso de TensorFlow con Smart Shopping Cart

Este script demuestra cómo:
1. Entrenar un modelo de clasificación con TensorFlow
2. Cargar un modelo entrenado
3. Usar el modelo para clasificar imágenes
"""

from object_detector import ObjectDetector
import cv2
import os

def main():
    print("=" * 60)
    print("EJEMPLO DE USO DE TENSORFLOW")
    print("=" * 60)
    
    # 1. Inicializar detector
    print("\n1. Inicializando detector...")
    detector = ObjectDetector()
    
    # 2. Verificar estado de TensorFlow
    print("\n2. Verificando estado de TensorFlow...")
    tf_status = detector.get_tensorflow_status()
    print(f"   TensorFlow disponible: {tf_status.get('available', False)}")
    
    if not tf_status.get('available', False):
        print("\n[ERROR] TensorFlow no está disponible.")
        print("   Instala TensorFlow: pip install tensorflow==2.13.0")
        return
    
    # 3. Verificar si hay productos para entrenar
    print("\n3. Verificando productos disponibles...")
    training_status = detector.get_training_status()
    product_count = training_status.get('custom_products_count', 0)
    print(f"   Productos encontrados: {product_count}")
    
    if product_count == 0:
        print("\n[ADVERTENCIA] No hay productos para entrenar.")
        print("   Primero captura imágenes de productos usando la interfaz web.")
        return
    
    # 4. Preguntar si entrenar
    print("\n4. ¿Deseas entrenar un nuevo modelo?")
    print("   (Esto puede tomar varios minutos)")
    respuesta = input("   Entrenar? (s/n): ").lower().strip()
    
    if respuesta == 's':
        print("\n   Iniciando entrenamiento...")
        print("   Esto puede tomar varios minutos, por favor espera...")
        
        result = detector.train_tensorflow_model(
            epochs=20,  # Reducido para ejemplo rápido
            batch_size=16,
            model_type='mobilenet'  # Rápido y eficiente
        )
        
        if result:
            print(f"\n[OK] Entrenamiento completado!")
            print(f"   Accuracy: {result['test_accuracy']:.4f}")
            print(f"   Top-K Accuracy: {result['test_top_k_accuracy']:.4f}")
            print(f"   Modelo guardado en: {result['model_path']}")
        else:
            print("\n[ERROR] Error en el entrenamiento")
            return
    
    # 5. Cargar modelo
    print("\n5. Cargando modelo TensorFlow...")
    if detector.load_tensorflow_model():
        print("   [OK] Modelo cargado correctamente")
        tf_status = detector.get_tensorflow_status()
        if tf_status.get('class_names'):
            print(f"   Clases disponibles: {', '.join(tf_status['class_names'])}")
    else:
        print("   [ADVERTENCIA] No se pudo cargar el modelo")
        print("   Intenta entrenar primero o verifica que existe un modelo guardado")
        return
    
    # 6. Probar clasificación
    print("\n6. Probar clasificación...")
    print("   Ingresa la ruta de una imagen para clasificar (o 'q' para salir):")
    
    while True:
        image_path = input("   Ruta de imagen: ").strip()
        
        if image_path.lower() == 'q':
            break
        
        if not os.path.exists(image_path):
            print(f"   [ERROR] Archivo no encontrado: {image_path}")
            continue
        
        # Cargar y clasificar imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"   [ERROR] No se pudo cargar la imagen: {image_path}")
            continue
        
        print(f"   Clasificando {image_path}...")
        detections = detector.detect_with_tensorflow(image)
        
        if detections:
            print(f"\n   Predicciones (top {len(detections)}):")
            for i, det in enumerate(detections, 1):
                print(f"   {i}. {det['product_name']}: {det['confidence']:.2%}")
        else:
            print("   No se encontraron predicciones con confianza suficiente")
    
    print("\n" + "=" * 60)
    print("FIN DEL EJEMPLO")
    print("=" * 60)

if __name__ == "__main__":
    main()

