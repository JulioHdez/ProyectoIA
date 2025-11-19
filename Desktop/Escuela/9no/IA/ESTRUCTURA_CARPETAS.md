# 📁 Estructura de Carpetas por Producto - Custom Training

## 🎯 Descripción

El sistema de entrenamiento personalizado ahora organiza cada producto en su propia carpeta con subcarpetas organizadas. Esto permite una mejor gestión, organización y escalabilidad del sistema.

## 📂 Estructura de Carpetas

```
custom_products/
├── Producto1/                    # Carpeta principal del producto
│   ├── images/                   # Imágenes originales capturadas
│   │   ├── Producto1_0_20251028_200437.jpg
│   │   ├── Producto1_1_20251028_200437.jpg
│   │   └── ...
│   ├── labels/                   # Etiquetas YOLO (opcional)
│   │   ├── Producto1_0_20251028_200437.txt
│   │   └── ...
│   ├── training/                 # Datos preparados para entrenamiento
│   │   ├── images/              # Imágenes copiadas para entrenamiento
│   │   └── labels/              # Etiquetas copiadas para entrenamiento
│   ├── models/                   # Modelos entrenados específicos del producto
│   │   ├── Producto1.pt        # Modelo individual del producto
│   │   └── ensemble_model.pt   # Modelo conjunto (si existe)
│   ├── data.yaml                # Configuración YOLO para este producto
│   └── metadata.json            # Información del producto
│
├── Producto2/                    # Otro producto
│   ├── images/
│   ├── labels/
│   ├── training/
│   ├── models/
│   ├── data.yaml
│   └── metadata.json
│
└── ...
```

## 🔍 Descripción de Subcarpetas

### 📸 `images/`
- **Contenido**: Imágenes originales capturadas del producto
- **Formato**: JPG, PNG, BMP
- **Uso**: Almacenamiento permanente de las imágenes de entrenamiento

### 🏷️ `labels/`
- **Contenido**: Etiquetas YOLO en formato `.txt`
- **Formato**: `class_id center_x center_y width height`
- **Uso**: Anotaciones manuales o generadas automáticamente

### 🎓 `training/`
- **Contenido**: Datos preparados específicamente para el entrenamiento
- **Subcarpetas**:
  - `images/`: Copias de imágenes listas para entrenar
  - `labels/`: Etiquetas correspondientes
- **Uso**: Datos procesados y listos para YOLO

### 🤖 `models/`
- **Contenido**: Modelos de IA entrenados
- **Tipos**:
  - `Producto.pt`: Modelo entrenado solo para este producto
  - `ensemble_model.pt`: Modelo conjunto (todos los productos)
- **Uso**: Modelos listos para detección

### 📄 `metadata.json`
- **Contenido**: Información del producto
- **Ejemplo**:
```json
{
  "name": "Pluma",
  "safe_name": "Pluma",
  "image_count": 10,
  "image_paths": [
    "custom_products/Pluma/images/Pluma_0_20251028_200437.jpg",
    ...
  ],
  "created_at": "2025-10-28T20:04:37.773785",
  "last_updated": "2025-10-28T20:04:37.773785",
  "base_dir": "custom_products/Pluma",
  "source": "web_capture"
}
```

### ⚙️ `data.yaml`
- **Contenido**: Configuración YOLO para entrenamiento
- **Ejemplo**:
```yaml
path: C:\Users\...\custom_products\Pluma\training
train: images
val: images

nc: 1
names: ['Pluma']
```

## 🚀 Ventajas de esta Estructura

### ✅ Organización
- Cada producto tiene su propio espacio
- Fácil de encontrar y gestionar
- No hay mezcla de archivos entre productos

### ✅ Escalabilidad
- Agregar nuevos productos no afecta los existentes
- Cada producto puede tener múltiples modelos
- Fácil de hacer backup de productos específicos

### ✅ Flexibilidad
- Entrenar productos individuales o en conjunto
- Modelos específicos por producto
- Metadata independiente por producto

### ✅ Mantenimiento
- Eliminar un producto es eliminar su carpeta
- Fácil de mover o copiar productos
- Estructura clara y predecible

## 🔧 Funciones Principales

### Crear Estructura de Producto
```python
trainer = CustomProductTrainer()
trainer._create_product_directories("Mi Producto")
# Crea: custom_products/Mi_Producto/ con todas las subcarpetas
```

### Agregar Producto
```python
trainer.add_product_for_training("Pluma", images)
# Guarda imágenes en: custom_products/Pluma/images/
# Crea metadata en: custom_products/Pluma/metadata.json
```

### Entrenar Producto Individual
```python
trainer.train_custom_model(epochs=50, product_name="Pluma")
# Entrena y guarda en: custom_products/Pluma/models/Pluma.pt
```

### Entrenar Todos los Productos
```python
trainer.train_custom_model(epochs=50)
# Entrena conjunto y guarda en cada carpeta: models/ensemble_model.pt
```

### Eliminar Producto
```python
trainer.delete_product("Pluma")
# Elimina completamente: custom_products/Pluma/
```

## 📝 Notas Importantes

1. **Nombres Sanitizados**: Los nombres de productos se sanitizan para usar como nombres de carpeta
   - Espacios → guiones bajos
   - Caracteres especiales → eliminados
   - Ejemplo: "Mi Producto!" → "Mi_Producto"

2. **Rutas Relativas**: Todas las rutas son relativas al directorio del proyecto

3. **Compatibilidad**: La estructura es compatible con YOLO estándar

4. **Backup**: Cada carpeta de producto es independiente y fácil de respaldar

## 🔄 Migración desde Estructura Anterior

Si tienes productos en la estructura antigua (`training_data/raw_images/`), el sistema los migrará automáticamente la próxima vez que agregues imágenes o entrenes.

## 📊 Ejemplo de Uso Completo

```python
# 1. Crear entrenador
trainer = CustomProductTrainer()

# 2. Agregar producto con imágenes
trainer.add_product_for_training("Laptop", images)
# → Crea: custom_products/Laptop/

# 3. Preparar datos de entrenamiento
trainer.prepare_training_data("Laptop")
# → Prepara: custom_products/Laptop/training/

# 4. Entrenar modelo específico
trainer.train_custom_model(epochs=50, product_name="Laptop")
# → Guarda: custom_products/Laptop/models/Laptop.pt

# 5. Agregar otro producto
trainer.add_product_for_training("Mouse", mouse_images)
# → Crea: custom_products/Mouse/

# 6. Entrenar modelo conjunto
trainer.train_custom_model(epochs=50)
# → Guarda modelos en cada carpeta: models/ensemble_model.pt
```

## 🎯 Resultado Final

Cada producto tiene su propia "casa" con:
- ✅ Sus propias imágenes
- ✅ Sus propias etiquetas
- ✅ Sus propios datos de entrenamiento
- ✅ Sus propios modelos entrenados
- ✅ Su propia metadata

¡Todo organizado y separado! 🎉
