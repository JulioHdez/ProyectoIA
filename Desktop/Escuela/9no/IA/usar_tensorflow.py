"""
Script para usar un modelo TensorFlow entrenado

Uso:
    python usar_tensorflow.py [ruta_imagen]
"""

from object_detector import ObjectDetector
import cv2
import sys
import os

def main():
    print("=" * 60)
    print("CLASIFICACIÓN CON TENSORFLOW")
    print("=" * 60)
    
    # 1. Inicializar detector
    print("\n[1/4] Inicializando detector...")
    detector = ObjectDetector()
    
    # 2. Verificar TensorFlow
    print("\n[2/4] Verificando TensorFlow...")
    tf_status = detector.get_tensorflow_status()
    
    if not tf_status.get('available', False):
        print("\n[ERROR] TensorFlow no está disponible")
        print("Instala con: pip install tensorflow==2.13.0")
        return
    
    print("  [OK] TensorFlow disponible")
    
    # 3. Cargar modelo
    print("\n[3/4] Cargando modelo TensorFlow...")
    
    # Intentar cargar modelo más reciente
    if detector.load_tensorflow_model():
        print("  [OK] Modelo cargado correctamente")
        
        # Mostrar información del modelo
        status = detector.get_tensorflow_status()
        if status.get('class_names'):
            print(f"  Clases disponibles: {', '.join(status['class_names'])}")
    else:
        print("\n[ERROR] No se pudo cargar el modelo")
        print("\nPosibles causas:")
        print("  1. No has entrenado un modelo aún")
        print("  2. El modelo no existe en tensorflow_models/saved_models/")
        print("\nSolución:")
        print("  Ejecuta primero: python entrenar_tensorflow.py")
        return
    
    # 4. Clasificar imagen
    print("\n[4/4] Clasificando imagen...")
    
    # Obtener ruta de imagen
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("  Ingresa la ruta de la imagen: ").strip()
    
    if not image_path:
        print("  [ERROR] No se proporcionó ruta de imagen")
        return
    
    if not os.path.exists(image_path):
        print(f"  [ERROR] Archivo no encontrado: {image_path}")
        return
    
    # Cargar imagen
    print(f"  Cargando: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"  [ERROR] No se pudo cargar la imagen")
        return
    
    print(f"  Imagen cargada: {image.shape[1]}x{image.shape[0]} píxeles")
    
    # Clasificar
    print("  Clasificando...")
    detections = detector.detect_with_tensorflow(image)
    
    # Mostrar resultados
    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    
    if detections:
        print(f"\nSe encontraron {len(detections)} predicción(es):\n")
        for i, det in enumerate(detections, 1):
            confidence_pct = det['confidence'] * 100
            print(f"  {i}. {det['product_name']}")
            print(f"     Confianza: {confidence_pct:.2f}%")
        
        # Mostrar la mejor predicción
        best = detections[0]
        print(f"\nMejor predicción: {best['product_name']} ({best['confidence']*100:.2f}%)")
    else:
        print("\nNo se encontraron predicciones con confianza suficiente")
        print("El modelo no está seguro sobre qué producto es")
        print("\nSugerencias:")
        print("  - Asegúrate de que la imagen muestre un producto conocido")
        print("  - Verifica que el producto esté bien iluminado")
        print("  - Intenta con otra imagen del mismo producto")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

