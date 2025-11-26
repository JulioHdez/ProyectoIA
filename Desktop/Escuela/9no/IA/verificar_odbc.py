"""
Script para verificar qué drivers ODBC para SQL Server tienes instalados

Uso:
    python verificar_odbc.py
"""
import pyodbc

def verificar_drivers_odbc():
    """Verifica qué drivers ODBC para SQL Server están instalados"""
    print("=" * 60)
    print("VERIFICACIÓN DE DRIVERS ODBC PARA SQL SERVER")
    print("=" * 60)
    
    try:
        # Obtener todos los drivers instalados
        drivers = pyodbc.drivers()
        
        # Filtrar drivers de SQL Server
        sql_drivers = [d for d in drivers if 'SQL Server' in d]
        
        if not sql_drivers:
            print("\n[ADVERTENCIA] No se encontraron drivers ODBC para SQL Server")
            print("\nNecesitas instalar un driver ODBC:")
            print("  1. Descarga desde: https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server")
            print("  2. Instala 'ODBC Driver 17 for SQL Server' o superior")
            print("  3. Ejecuta este script nuevamente para verificar")
            return None
        
        print(f"\n[OK] Encontrados {len(sql_drivers)} driver(s) de SQL Server:\n")
        
        for i, driver in enumerate(sql_drivers, 1):
            print(f"  {i}. {driver}")
        
        # Recomendar el mejor driver
        recommended_drivers = [
            'ODBC Driver 18 for SQL Server',
            'ODBC Driver 17 for SQL Server',
            'ODBC Driver 13 for SQL Server'
        ]
        
        recommended = None
        for rec in recommended_drivers:
            if rec in sql_drivers:
                recommended = rec
                break
        
        if recommended:
            print(f"\n[RECOMENDADO] Usa este driver en config.py:")
            print(f"  '{recommended}'")
            print(f"\nCadena de conexión recomendada:")
            print(f"  SQLALCHEMY_DATABASE_URI = 'mssql+pyodbc://localhost/smart_cart?driver={recommended.replace(' ', '+')}&trusted_connection=yes'")
        else:
            # Usar el primero disponible
            driver_name = sql_drivers[0]
            print(f"\n[SUGERENCIA] Usa este driver en config.py:")
            print(f"  '{driver_name}'")
            print(f"\nCadena de conexión sugerida:")
            # Reemplazar espacios con + para URL
            driver_encoded = driver_name.replace(' ', '+')
            print(f"  SQLALCHEMY_DATABASE_URI = 'mssql+pyodbc://localhost/smart_cart?driver={driver_encoded}&trusted_connection=yes'")
        
        print("\n" + "=" * 60)
        print("INSTRUCCIONES")
        print("=" * 60)
        print("\n1. Copia la cadena de conexión de arriba")
        print("2. Pégala en config.py reemplazando SQLALCHEMY_DATABASE_URI")
        print("3. Asegúrate de que IMAGE_STORAGE_METHOD = 'database'")
        print("4. Ejecuta: python app.py")
        
        return recommended or sql_drivers[0]
        
    except ImportError:
        print("\n[ERROR] pyodbc no está instalado")
        print("\nInstala con:")
        print("  pip install pyodbc==5.0.1")
        return None
    except Exception as e:
        print(f"\n[ERROR] Error al verificar drivers: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    verificar_drivers_odbc()

