"""
Script simple para entrenar un modelo TensorFlow

Uso:
    python entrenar_tensorflow.py
"""

from object_detector import ObjectDetector

def main():
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELO TENSORFLOW")
    print("=" * 60)
    
    # 1. Inicializar detector
    print("\n[1/5] Inicializando detector...")
    detector = ObjectDetector()
    
    # 2. Verificar TensorFlow
    print("\n[2/5] Verificando TensorFlow...")
    tf_status = detector.get_tensorflow_status()
    
    if not tf_status.get('available', False):
        print("\n[ERROR] TensorFlow no está disponible")
        print("\nInstala TensorFlow con:")
        print("  pip install tensorflow==2.13.0")
        return
    
    print("  [OK] TensorFlow está disponible")
    
    # 3. Verificar productos
    print("\n[3/5] Verificando productos disponibles...")
    training_status = detector.get_training_status()
    product_count = training_status.get('custom_products_count', 0)
    
    print(f"  Productos encontrados: {product_count}")
    
    if product_count < 2:
        print("\n[ADVERTENCIA] Necesitas al menos 2 productos para entrenar")
        print("\nOpciones:")
        print("  1. Ve a http://localhost:5000/training y captura imágenes")
        print("  2. O usa: detector.capture_product_images('Producto', 0, 20)")
        return
    
    products = training_status.get('products', [])
    print(f"  Productos: {', '.join(products)}")
    
    # 4. Configurar entrenamiento
    print("\n[4/5] Configurando entrenamiento...")
    
    # Preguntar parámetros (opcional)
    print("\nParámetros de entrenamiento:")
    print("  - Épocas: 50 (recomendado)")
    print("  - Batch size: 32 (recomendado)")
    print("  - Modelo: mobilenet (rápido y eficiente)")
    
    usar_defaults = input("\n¿Usar parámetros por defecto? (s/n): ").lower().strip()
    
    if usar_defaults == 's' or usar_defaults == '':
        epochs = 50
        batch_size = 32
        model_type = 'mobilenet'
    else:
        try:
            epochs = int(input("  Épocas (50): ") or "50")
            batch_size = int(input("  Batch size (32): ") or "32")
            print("  Tipos disponibles: mobilenet, efficientnet, resnet, custom")
            model_type = input("  Tipo de modelo (mobilenet): ").strip() or "mobilenet"
        except ValueError:
            print("  Usando valores por defecto...")
            epochs = 50
            batch_size = 32
            model_type = 'mobilenet'
    
    print(f"\n  Configuración:")
    print(f"    - Épocas: {epochs}")
    print(f"    - Batch size: {batch_size}")
    print(f"    - Modelo: {model_type}")
    
    # 5. Entrenar
    print("\n[5/5] Iniciando entrenamiento...")
    print("  Esto puede tomar varios minutos, por favor espera...")
    print("  (Puedes ver el progreso arriba)\n")
    
    result = detector.train_tensorflow_model(
        epochs=epochs,
        batch_size=batch_size,
        model_type=model_type
    )
    
    # Resultados
    print("\n" + "=" * 60)
    if result:
        print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print(f"\nResultados:")
        print(f"  Accuracy: {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)")
        print(f"  Top-K Accuracy: {result['test_top_k_accuracy']:.4f}")
        print(f"\nModelo guardado en:")
        print(f"  {result['model_path']}")
        print(f"\nPara usar el modelo:")
        print(f"  detector.load_tensorflow_model()")
        print(f"  detections = detector.detect_with_tensorflow(image)")
    else:
        print("ERROR EN EL ENTRENAMIENTO")
        print("=" * 60)
        print("\nRevisa los mensajes de error arriba")
        print("Asegúrate de tener suficientes imágenes de cada producto")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

