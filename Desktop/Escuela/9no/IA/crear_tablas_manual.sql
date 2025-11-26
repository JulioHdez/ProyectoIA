-- Script SQL para crear las tablas manualmente en SQL Server
-- Ejecuta este script en SQL Server Management Studio (SSMS)

USE smart_cart;
GO

-- Tabla: products
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'products')
BEGIN
    CREATE TABLE products (
        id INT PRIMARY KEY IDENTITY(1,1),
        name NVARCHAR(100) NOT NULL,
        description NVARCHAR(MAX),
        price FLOAT NOT NULL,
        category NVARCHAR(50),
        barcode NVARCHAR(50) UNIQUE,
        image_path NVARCHAR(200),
        detection_method NVARCHAR(20) DEFAULT 'manual',
        created_at DATETIME DEFAULT GETDATE()
    );
    PRINT 'Tabla products creada';
END
ELSE
    PRINT 'Tabla products ya existe';
GO

-- Tabla: inventory
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'inventory')
BEGIN
    CREATE TABLE inventory (
        id INT PRIMARY KEY IDENTITY(1,1),
        product_id INT NOT NULL,
        quantity INT NOT NULL DEFAULT 0,
        min_stock INT DEFAULT 5,
        last_updated DATETIME DEFAULT GETDATE(),
        FOREIGN KEY (product_id) REFERENCES products(id)
    );
    PRINT 'Tabla inventory creada';
END
ELSE
    PRINT 'Tabla inventory ya existe';
GO

-- Tabla: sales
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'sales')
BEGIN
    CREATE TABLE sales (
        id INT PRIMARY KEY IDENTITY(1,1),
        product_id INT NOT NULL,
        quantity INT NOT NULL,
        unit_price FLOAT NOT NULL,
        total_price FLOAT NOT NULL,
        sale_date DATE DEFAULT CAST(GETDATE() AS DATE),
        sale_time DATETIME DEFAULT GETDATE(),
        detection_method NVARCHAR(20) DEFAULT 'manual',
        FOREIGN KEY (product_id) REFERENCES products(id)
    );
    PRINT 'Tabla sales creada';
END
ELSE
    PRINT 'Tabla sales ya existe';
GO

-- Tabla: daily_reports
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'daily_reports')
BEGIN
    CREATE TABLE daily_reports (
        id INT PRIMARY KEY IDENTITY(1,1),
        report_date DATE UNIQUE NOT NULL,
        total_sales FLOAT DEFAULT 0.0,
        total_products_sold INT DEFAULT 0,
        total_transactions INT DEFAULT 0,
        created_at DATETIME DEFAULT GETDATE()
    );
    PRINT 'Tabla daily_reports creada';
END
ELSE
    PRINT 'Tabla daily_reports ya existe';
GO

-- Tabla: weekly_reports
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'weekly_reports')
BEGIN
    CREATE TABLE weekly_reports (
        id INT PRIMARY KEY IDENTITY(1,1),
        week_start DATE NOT NULL,
        week_end DATE NOT NULL,
        total_sales FLOAT DEFAULT 0.0,
        total_products_sold INT DEFAULT 0,
        total_transactions INT DEFAULT 0,
        top_selling_products NVARCHAR(MAX),
        created_at DATETIME DEFAULT GETDATE()
    );
    PRINT 'Tabla weekly_reports creada';
END
ELSE
    PRINT 'Tabla weekly_reports ya existe';
GO

-- Tabla: training_products
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'training_products')
BEGIN
    CREATE TABLE training_products (
        id INT PRIMARY KEY IDENTITY(1,1),
        name NVARCHAR(100) NOT NULL UNIQUE,
        created_at DATETIME DEFAULT GETDATE(),
        last_updated DATETIME DEFAULT GETDATE()
    );
    PRINT 'Tabla training_products creada';
END
ELSE
    PRINT 'Tabla training_products ya existe';
GO

-- Tabla: training_images
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'training_images')
BEGIN
    CREATE TABLE training_images (
        id INT PRIMARY KEY IDENTITY(1,1),
        product_id INT NOT NULL,
        filename NVARCHAR(255) NOT NULL,
        image_data VARBINARY(MAX) NOT NULL,
        image_format NVARCHAR(10) DEFAULT 'JPEG',
        file_size INT,
        created_at DATETIME DEFAULT GETDATE(),
        FOREIGN KEY (product_id) REFERENCES training_products(id) ON DELETE CASCADE
    );
    PRINT 'Tabla training_images creada';
END
ELSE
    PRINT 'Tabla training_images ya existe';
GO

-- Verificar tablas creadas
PRINT '';
PRINT '========================================';
PRINT 'VERIFICACIÓN DE TABLAS';
PRINT '========================================';
SELECT 
    TABLE_NAME as 'Tabla',
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = t.TABLE_NAME) as 'Columnas'
FROM INFORMATION_SCHEMA.TABLES t
WHERE TABLE_TYPE = 'BASE TABLE'
ORDER BY TABLE_NAME;
GO

PRINT '';
PRINT 'Todas las tablas han sido creadas/verificadas.';
GO

