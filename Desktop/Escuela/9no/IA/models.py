from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date
from sqlalchemy import func

db = SQLAlchemy()

class Product(db.Model):
    """Modelo para productos del inventario"""
    __tablename__ = 'products'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    price = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50))
    barcode = db.Column(db.String(50), unique=True)
    image_path = db.Column(db.String(200))
    detection_method = db.Column(db.String(20), default='manual')  # 'camera' o 'manual'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relación con inventario
    inventory = db.relationship('Inventory', backref='product', lazy=True)
    # Relación con ventas
    sales = db.relationship('Sale', backref='product', lazy=True)
    
    def __repr__(self):
        return f'<Product {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'price': self.price,
            'category': self.category,
            'barcode': self.barcode,
            'image_path': self.image_path,
            'detection_method': self.detection_method,
            'created_at': self.created_at.isoformat()
        }

class Inventory(db.Model):
    """Modelo para control de inventario"""
    __tablename__ = 'inventory'
    
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=0)
    min_stock = db.Column(db.Integer, default=5)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Inventory {self.product.name}: {self.quantity}>'
    
    def is_low_stock(self):
        return self.quantity <= self.min_stock
    
    def to_dict(self):
        return {
            'id': self.id,
            'product_id': self.product_id,
            'product_name': self.product.name,
            'quantity': self.quantity,
            'min_stock': self.min_stock,
            'is_low_stock': self.is_low_stock(),
            'last_updated': self.last_updated.isoformat()
        }

class Sale(db.Model):
    """Modelo para registro de ventas"""
    __tablename__ = 'sales'
    
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    unit_price = db.Column(db.Float, nullable=False)
    total_price = db.Column(db.Float, nullable=False)
    sale_date = db.Column(db.Date, default=date.today)
    sale_time = db.Column(db.DateTime, default=datetime.utcnow)
    detection_method = db.Column(db.String(20), default='manual')  # 'camera' o 'manual'
    
    def __repr__(self):
        return f'<Sale {self.product.name}: {self.quantity} x ${self.unit_price}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'product_id': self.product_id,
            'product_name': self.product.name,
            'quantity': self.quantity,
            'unit_price': self.unit_price,
            'total_price': self.total_price,
            'sale_date': self.sale_date.isoformat(),
            'sale_time': self.sale_time.isoformat(),
            'detection_method': self.detection_method
        }

class DailyReport(db.Model):
    """Modelo para reportes diarios"""
    __tablename__ = 'daily_reports'
    
    id = db.Column(db.Integer, primary_key=True)
    report_date = db.Column(db.Date, unique=True, nullable=False)
    total_sales = db.Column(db.Float, default=0.0)
    total_products_sold = db.Column(db.Integer, default=0)
    total_transactions = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<DailyReport {self.report_date}: ${self.total_sales}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'report_date': self.report_date.isoformat(),
            'total_sales': self.total_sales,
            'total_products_sold': self.total_products_sold,
            'total_transactions': self.total_transactions,
            'created_at': self.created_at.isoformat()
        }

class WeeklyReport(db.Model):
    """Modelo para reportes semanales"""
    __tablename__ = 'weekly_reports'
    
    id = db.Column(db.Integer, primary_key=True)
    week_start = db.Column(db.Date, nullable=False)
    week_end = db.Column(db.Date, nullable=False)
    total_sales = db.Column(db.Float, default=0.0)
    total_products_sold = db.Column(db.Integer, default=0)
    total_transactions = db.Column(db.Integer, default=0)
    top_selling_products = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<WeeklyReport {self.week_start} - {self.week_end}: ${self.total_sales}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'week_start': self.week_start.isoformat(),
            'week_end': self.week_end.isoformat(),
            'total_sales': self.total_sales,
            'total_products_sold': self.total_products_sold,
            'total_transactions': self.total_transactions,
            'top_selling_products': self.top_selling_products,
            'created_at': self.created_at.isoformat()
        }

# Funciones de utilidad para reportes
def get_daily_sales_stats(target_date=None):
    """Obtiene estadísticas de ventas para un día específico"""
    if target_date is None:
        target_date = date.today()
    
    sales_query = Sale.query.filter(Sale.sale_date == target_date)
    
    total_sales = sales_query.with_entities(func.sum(Sale.total_price)).scalar() or 0
    total_products = sales_query.with_entities(func.sum(Sale.quantity)).scalar() or 0
    total_transactions = sales_query.count()
    
    return {
        'date': target_date,
        'total_sales': float(total_sales),
        'total_products_sold': int(total_products),
        'total_transactions': total_transactions
    }

def get_weekly_sales_stats(start_date, end_date):
    """Obtiene estadísticas de ventas para un rango de fechas"""
    sales_query = Sale.query.filter(
        Sale.sale_date >= start_date,
        Sale.sale_date <= end_date
    )
    
    total_sales = sales_query.with_entities(func.sum(Sale.total_price)).scalar() or 0
    total_products = sales_query.with_entities(func.sum(Sale.quantity)).scalar() or 0
    total_transactions = sales_query.count()
    
    # Productos más vendidos
    top_products = db.session.query(
        Product.name,
        func.sum(Sale.quantity).label('total_sold')
    ).join(Sale).filter(
        Sale.sale_date >= start_date,
        Sale.sale_date <= end_date
    ).group_by(Product.id, Product.name).order_by(
        func.sum(Sale.quantity).desc()
    ).limit(5).all()
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'total_sales': float(total_sales),
        'total_products_sold': int(total_products),
        'total_transactions': total_transactions,
        'top_products': [{'name': p.name, 'quantity': p.total_sold} for p in top_products]
    }

