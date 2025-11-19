from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
import cv2
import base64
import json
from datetime import datetime, date, timedelta
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import threading
import time

from models import db, Product, Inventory, Sale, DailyReport, WeeklyReport, get_daily_sales_stats, get_weekly_sales_stats
from object_detector import ObjectDetector, CameraHandler

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tu_clave_secreta_aqui'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///smart_cart.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configurar ruta estática para productos personalizados
app.static_folder = 'static'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Inicializar extensiones
db.init_app(app)
migrate = Migrate(app, db)

# Inicializar detector de objetos
detector = ObjectDetector()
camera_handler = CameraHandler()

# Variables globales para streaming de cámara
camera_streaming = False
detection_results = []

# Las tablas se crean automáticamente en la función main

@app.route('/')
def welcome():
    """Pantalla de inicio - Selección de modo"""
    return render_template('welcome.html')

@app.route('/admin')
def admin_dashboard():
    """Panel de administración"""
    # Estadísticas rápidas
    total_products = Product.query.count()
    low_stock_items = Inventory.query.filter(Inventory.quantity <= Inventory.min_stock).count()
    today_sales = Sale.query.filter(Sale.sale_date == date.today()).count()
    today_revenue = db.session.query(db.func.sum(Sale.total_price)).filter(
        Sale.sale_date == date.today()
    ).scalar() or 0
    
    stats = {
        'total_products': total_products,
        'low_stock_items': low_stock_items,
        'today_sales': today_sales,
        'today_revenue': float(today_revenue)
    }
    
    return render_template('index.html', stats=stats)

@app.route('/checkout')
def checkout():
    """Página de cobro con cámara y carrito"""
    return render_template('checkout.html')

@app.route('/process_checkout', methods=['POST'])
def process_checkout():
    """Procesar venta del carrito y actualizar inventario"""
    try:
        data = request.get_json()
        items = data.get('items', [])
        total = data.get('total', 0)
        
        if not items or len(items) == 0:
            return jsonify({'status': 'error', 'message': 'El carrito está vacío'})
        
        warnings = []
        processed_items = []
        
        # Procesar cada item
        for item in items:
            # Buscar o crear producto
            product = Product.query.filter_by(name=item['name']).first()
            
            if not product:
                # Crear producto automáticamente
                product = Product(
                    name=item['name'],
                    description=f'Producto vendido desde carrito (confianza: {item.get("confidence", 0):.2f})',
                    price=item['price'],
                    category='Venta Rápida'
                )
                db.session.add(product)
                db.session.flush()
                
                # Crear inventario inicial (se asume que se vende desde stock existente)
                # Si no hay inventario previo, se crea con cantidad 0 y se permite la venta
                inventory = Inventory(
                    product_id=product.id,
                    quantity=0,
                    min_stock=1
                )
                db.session.add(inventory)
                db.session.flush()
                warnings.append(f'{item["name"]}: Producto nuevo creado (sin stock previo)')
            
            # Obtener o crear inventario
            inventory = Inventory.query.filter_by(product_id=product.id).first()
            if not inventory:
                # Si por alguna razón no existe inventario, crearlo
                inventory = Inventory(
                    product_id=product.id,
                    quantity=0,
                    min_stock=1
                )
                db.session.add(inventory)
                db.session.flush()
            
            # Verificar stock disponible
            current_stock = inventory.quantity
            requested_quantity = item['quantity']
            
            if current_stock < requested_quantity:
                warnings.append(
                    f'{item["name"]}: Stock insuficiente. '
                    f'Disponible: {current_stock}, Solicitado: {requested_quantity}. '
                    f'Se procesará la venta y el inventario quedará en {max(0, current_stock - requested_quantity)}'
                )
            
            # Registrar venta
            sale = Sale(
                product_id=product.id,
                quantity=requested_quantity,
                unit_price=item['price'],
                total_price=item['total'],
                detection_method='checkout'
            )
            db.session.add(sale)
            
            # Actualizar inventario: restar la cantidad vendida
            new_quantity = max(0, current_stock - requested_quantity)
            inventory.quantity = new_quantity
            inventory.last_updated = datetime.utcnow()
            
            processed_items.append({
                'name': item['name'],
                'quantity_sold': requested_quantity,
                'stock_before': current_stock,
                'stock_after': new_quantity
            })
        
        db.session.commit()
        
        response = {
            'status': 'success',
            'message': f'Venta procesada exitosamente. Total: ${total:.2f}',
            'total': total,
            'items_count': len(items),
            'processed_items': processed_items
        }
        
        if warnings:
            response['warnings'] = warnings
        
        return jsonify(response)
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/products')
def products():
    """Página de gestión de productos"""
    products = Product.query.all()
    return render_template('products.html', products=products)

@app.route('/add_product', methods=['GET', 'POST'])
def add_product():
    """Agregar nuevo producto con validaciones"""
    if request.method == 'POST':
        try:
            # Validar y obtener nombre
            name = request.form.get('name', '').strip()
            if not name or len(name) < 2:
                flash('El nombre del producto debe tener al menos 2 caracteres', 'error')
                return render_template('add_product.html')
            if len(name) > 100:
                flash('El nombre del producto no puede exceder 100 caracteres', 'error')
                return render_template('add_product.html')
            
            # Validar y obtener precio
            try:
                price_str = request.form.get('price', '0').replace(',', '.')
                price = float(price_str)
                if price < 0:
                    flash('El precio no puede ser negativo', 'error')
                    return render_template('add_product.html')
                if price > 999999.99:
                    flash('El precio no puede exceder $999,999.99', 'error')
                    return render_template('add_product.html')
            except (ValueError, TypeError):
                flash('El precio debe ser un número válido', 'error')
                return render_template('add_product.html')
            
            # Validar descripción
            description = request.form.get('description', '').strip()
            if len(description) > 500:
                flash('La descripción no puede exceder 500 caracteres', 'error')
                return render_template('add_product.html')
            
            # Validar categoría
            category = request.form.get('category', '').strip()
            if category and len(category) > 50:
                flash('La categoría no puede exceder 50 caracteres', 'error')
                return render_template('add_product.html')
            
            # Validar código de barras
            barcode = request.form.get('barcode', '').strip()
            if barcode:
                if len(barcode) > 50:
                    flash('El código de barras no puede exceder 50 caracteres', 'error')
                    return render_template('add_product.html')
                if not barcode.replace(' ', '').isalnum():
                    flash('El código de barras solo puede contener números y letras', 'error')
                    return render_template('add_product.html')
                # Verificar si el código de barras ya existe
                existing = Product.query.filter_by(barcode=barcode).first()
                if existing:
                    flash(f'El código de barras {barcode} ya está en uso', 'error')
                    return render_template('add_product.html')
            
            # Validar cantidad
            try:
                quantity = int(request.form.get('quantity', 0))
                if quantity < 0:
                    flash('La cantidad no puede ser negativa', 'error')
                    return render_template('add_product.html')
                if quantity > 999999:
                    flash('La cantidad no puede exceder 999,999', 'error')
                    return render_template('add_product.html')
            except (ValueError, TypeError):
                flash('La cantidad debe ser un número entero válido', 'error')
                return render_template('add_product.html')
            
            # Validar stock mínimo
            try:
                min_stock = int(request.form.get('min_stock', 5))
                if min_stock < 0:
                    flash('El stock mínimo no puede ser negativo', 'error')
                    return render_template('add_product.html')
                if min_stock > 999999:
                    flash('El stock mínimo no puede exceder 999,999', 'error')
                    return render_template('add_product.html')
            except (ValueError, TypeError):
                flash('El stock mínimo debe ser un número entero válido', 'error')
                return render_template('add_product.html')
            
            # Crear producto
            product = Product(
                name=name,
                description=description,
                price=price,
                category=category if category else None,
                barcode=barcode if barcode else None
            )
            
            db.session.add(product)
            db.session.flush()  # Para obtener el ID
            
            # Crear entrada de inventario
            inventory = Inventory(
                product_id=product.id,
                quantity=quantity,
                min_stock=min_stock
            )
            
            db.session.add(inventory)
            db.session.commit()
            
            flash('Producto agregado exitosamente', 'success')
            return redirect(url_for('products'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error al agregar producto: {str(e)}', 'error')
    
    return render_template('add_product.html')

@app.route('/inventory')
def inventory():
    """Página de inventario"""
    inventory_items = db.session.query(Inventory, Product).join(Product).all()
    return render_template('inventory.html', inventory_items=inventory_items)

@app.route('/update_inventory/<int:product_id>', methods=['POST'])
def update_inventory(product_id):
    """Actualizar cantidad en inventario con validaciones"""
    try:
        # Validar cantidad
        try:
            quantity_str = request.form.get('quantity', '0').strip()
            quantity = int(quantity_str)
            
            if quantity < 0:
                flash('La cantidad no puede ser negativa', 'error')
                return redirect(url_for('inventory'))
            
            if quantity > 999999:
                flash('La cantidad no puede exceder 999,999', 'error')
                return redirect(url_for('inventory'))
                
        except (ValueError, TypeError):
            flash('La cantidad debe ser un número entero válido', 'error')
            return redirect(url_for('inventory'))
        
        # Buscar inventario
        inventory = Inventory.query.filter_by(product_id=product_id).first()
        
        if inventory:
            inventory.quantity = quantity
            inventory.last_updated = datetime.utcnow()
            db.session.commit()
            flash('Inventario actualizado exitosamente', 'success')
        else:
            flash('Producto no encontrado en inventario', 'error')
            
    except Exception as e:
        db.session.rollback()
        flash(f'Error al actualizar inventario: {str(e)}', 'error')
    
    return redirect(url_for('inventory'))

@app.route('/camera')
def camera():
    """Página de detección por cámara"""
    return render_template('camera.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Iniciar detección de objetos"""
    global camera_streaming, detection_results
    
    try:
        camera_handler.start_camera()
        camera_streaming = True
        detection_results = []
        
        # Iniciar hilo de detección
        detection_thread = threading.Thread(target=detection_worker)
        detection_thread.daemon = True
        detection_thread.start()
        
        return jsonify({'status': 'success', 'message': 'Detección iniciada'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Detener detección de objetos"""
    global camera_streaming
    
    camera_streaming = False
    camera_handler.stop_camera()
    
    return jsonify({'status': 'success', 'message': 'Detección detenida'})

@app.route('/set_optimization_level', methods=['POST'])
def set_optimization_level():
    """Configurar nivel de optimización de la IA"""
    try:
        data = request.get_json()
        level = data.get('level', 'balanced')
        
        if level not in ['fast', 'balanced', 'accurate']:
            return jsonify({'status': 'error', 'message': 'Nivel inválido'})
        
        detector.set_optimization_level(level)
        
        return jsonify({
            'status': 'success',
            'message': f'Nivel de optimización configurado a: {level}',
            'level': level,
            'config': {
                'resolution': detector.process_resolution,
                'frame_skip': detector.frame_skip,
                'threshold': detector.detection_threshold,
                'device': detector.device
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_detection_results')
def get_detection_results():
    """Obtener resultados de detección en tiempo real (optimizado)"""
    global detection_results
    
    if camera_streaming:
        frame = camera_handler.get_frame()
        if frame is not None:
            # Usar cache para frames intermedios (frame skipping)
            detections = detector.detect_objects(frame, use_cache=True)
            annotated_frame = detector.draw_detections(frame, detections)
            
            # Optimizar compresión de imagen para menor tamaño
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]  # Calidad reducida para menor tamaño
            _, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'frame': frame_base64,
                'detections': detections,
                'suggestions': detector.get_product_suggestions(detections)
            })
    
    return jsonify({'frame': None, 'detections': [], 'suggestions': []})

@app.route('/add_detected_product', methods=['POST'])
def add_detected_product():
    """Agregar producto detectado al carrito"""
    try:
        data = request.get_json()
        product_name = data['product_name']
        confidence = data['confidence']
        suggested_price = data.get('suggested_price', 10.0)
        
        # Buscar producto existente
        product = Product.query.filter_by(name=product_name).first()
        
        if not product:
            # Crear nuevo producto
            product = Product(
                name=product_name,
                description=f'Producto detectado automáticamente (confianza: {confidence:.2f})',
                price=suggested_price,
                category='Detectado',
                detection_method='camera'
            )
            db.session.add(product)
            db.session.flush()
            
            # Crear entrada de inventario
            inventory = Inventory(
                product_id=product.id,
                quantity=1,
                min_stock=1
            )
            db.session.add(inventory)
        
        # Registrar venta
        sale = Sale(
            product_id=product.id,
            quantity=1,
            unit_price=product.price,
            total_price=product.price,
            detection_method='camera'
        )
        
        db.session.add(sale)
        
        # Actualizar inventario: restar la cantidad vendida
        inventory = Inventory.query.filter_by(product_id=product.id).first()
        if inventory:
            # Restar 1 del inventario (cantidad vendida)
            inventory.quantity = max(0, inventory.quantity - 1)
            inventory.last_updated = datetime.utcnow()
        else:
            # Si no existe inventario, crearlo con cantidad 0 (ya se vendió)
            inventory = Inventory(
                product_id=product.id,
                quantity=0,
                min_stock=1
            )
            db.session.add(inventory)
        
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': f'Producto {product_name} agregado al carrito',
            'product': product.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/sales')
def sales():
    """Página de ventas"""
    today = date.today()
    sales_today = Sale.query.filter(Sale.sale_date == today).all()
    
    # Estadísticas del día
    stats = get_daily_sales_stats(today)
    
    return render_template('sales.html', sales=sales_today, stats=stats)

@app.route('/reports')
def reports():
    """Página de reportes"""
    # Reportes diarios de la última semana
    week_ago = date.today() - timedelta(days=7)
    daily_reports = DailyReport.query.filter(
        DailyReport.report_date >= week_ago
    ).order_by(DailyReport.report_date.desc()).all()
    
    # Reportes semanales
    weekly_reports = WeeklyReport.query.order_by(
        WeeklyReport.week_start.desc()
    ).limit(4).all()
    
    return render_template('reports.html', 
                         daily_reports=daily_reports,
                         weekly_reports=weekly_reports)

@app.route('/generate_weekly_report', methods=['POST'])
def generate_weekly_report():
    """Generar reporte semanal"""
    try:
        # Calcular rango de la semana pasada
        today = date.today()
        week_start = today - timedelta(days=today.weekday() + 7)
        week_end = week_start + timedelta(days=6)
        
        # Obtener estadísticas
        stats = get_weekly_sales_stats(week_start, week_end)
        
        # Crear reporte semanal
        weekly_report = WeeklyReport(
            week_start=week_start,
            week_end=week_end,
            total_sales=stats['total_sales'],
            total_products_sold=stats['total_products_sold'],
            total_transactions=stats['total_transactions'],
            top_selling_products=json.dumps(stats['top_products'])
        )
        
        db.session.add(weekly_report)
        db.session.commit()
        
        flash('Reporte semanal generado exitosamente', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error al generar reporte: {str(e)}', 'error')
    
    return redirect(url_for('reports'))

@app.route('/training')
def training():
    """Página de entrenamiento personalizado"""
    training_status = detector.get_training_status()
    return render_template('training.html', status=training_status)

@app.route('/start_training', methods=['POST'])
def start_training():
    """Iniciar entrenamiento del modelo personalizado"""
    try:
        data = request.get_json()
        epochs = data.get('epochs', 50)
        batch_size = data.get('batch_size', 16)
        
        # Iniciar entrenamiento en un hilo separado
        training_thread = threading.Thread(
            target=detector.train_custom_model,
            args=(epochs, batch_size)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Entrenamiento iniciado en segundo plano'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/capture_product_images', methods=['POST'])
def capture_product_images():
    """Capturar imágenes de un producto para entrenamiento"""
    try:
        data = request.get_json()
        product_name = data['product_name']
        num_images = data.get('num_images', 10)
        
        # Iniciar captura en un hilo separado
        capture_thread = threading.Thread(
            target=detector.capture_product_images,
            args=(product_name, 0, num_images)
        )
        capture_thread.daemon = True
        capture_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': f'Captura iniciada para {product_name}'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_training_status')
def get_training_status():
    """Obtener estado del entrenamiento"""
    try:
        status = detector.get_training_status()
        return jsonify({'status': 'success', 'data': status})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_custom_products')
def get_custom_products():
    """Obtener información detallada de productos personalizados"""
    try:
        products = detector.custom_trainer.get_detailed_products()
        return jsonify({'status': 'success', 'products': products})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_product_images/<product_name>')
def get_product_images(product_name):
    """Obtener las imágenes de un producto específico"""
    try:
        images = detector.custom_trainer.get_product_images(product_name)
        return jsonify({'status': 'success', 'images': images})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/static/custom_products/<product_name>/<filename>')
def serve_custom_product_image(product_name, filename):
    """Servir imágenes de productos personalizados"""
    try:
        import os
        from flask import send_from_directory
        
        # Construir la ruta del archivo
        custom_data_dir = detector.custom_trainer.custom_data_dir
        product_dir = os.path.join(custom_data_dir, product_name)
        file_path = os.path.join(product_dir, filename)
        
        # Verificar que el archivo existe y está dentro del directorio permitido
        if os.path.exists(file_path) and file_path.startswith(custom_data_dir):
            return send_from_directory(product_dir, filename)
        else:
            return "Archivo no encontrado", 404
    except Exception as e:
        return f"Error al servir imagen: {str(e)}", 500

@app.route('/capture_images_from_web', methods=['POST'])
def capture_images_from_web():
    """Capturar imágenes desde la interfaz web"""
    try:
        data = request.get_json()
        product_name = data['product_name']
        images = data['images']
        num_images = data.get('num_images', len(images))
        
        # Usar el método de captura desde web
        success = detector.custom_trainer.capture_images_from_web(product_name, images, num_images)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Producto {product_name} agregado con {len(images)} imágenes'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Error al procesar las imágenes'
            })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/delete_custom_product', methods=['POST'])
def delete_custom_product():
    """Eliminar producto personalizado"""
    try:
        data = request.get_json()
        product_name = data['product_name']
        
        success = detector.delete_custom_product(product_name)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Producto {product_name} eliminado'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'No se pudo eliminar el producto {product_name}'
            })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/export_report/<int:report_id>')
def export_report(report_id):
    """Exportar reporte a CSV"""
    report = WeeklyReport.query.get_or_404(report_id)
    
    # Crear DataFrame con datos del reporte
    data = {
        'Fecha Inicio': [report.week_start],
        'Fecha Fin': [report.week_end],
        'Ventas Totales': [report.total_sales],
        'Productos Vendidos': [report.total_products_sold],
        'Transacciones': [report.total_transactions]
    }
    
    df = pd.DataFrame(data)
    
    # Crear archivo CSV en memoria
    output = BytesIO()
    df.to_csv(output, index=False, encoding='utf-8')
    output.seek(0)
    
    filename = f'reporte_semanal_{report.week_start}_{report.week_end}.csv'
    
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

def detection_worker():
    """Worker thread para detección continua (optimizado)"""
    global camera_streaming, detection_results
    
    while camera_streaming:
        try:
            frame = camera_handler.get_frame()
            if frame is not None:
                # Usar cache para mejor rendimiento
                detections = detector.detect_objects(frame, use_cache=True)
                detection_results = detections
            # Ajustar sleep según frame_skip para mejor rendimiento
            sleep_time = 0.1 / detector.frame_skip if detector.frame_skip > 1 else 0.1
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Error en worker de detección: {e}")
            break

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    app.run(debug=True, host='0.0.0.0', port=5000)

