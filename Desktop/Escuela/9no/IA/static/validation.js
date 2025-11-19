/**
 * Sistema de Validación para Smart Shopping Cart
 * Validaciones para formularios con tema oscuro
 */

// Validación para campos numéricos (solo números)
function validateNumericInput(event) {
    const input = event.target;
    const value = input.value;
    
    // Permitir teclas de control (backspace, delete, tab, etc.)
    if (event.key === 'Backspace' || event.key === 'Delete' || event.key === 'Tab' || 
        event.key === 'ArrowLeft' || event.key === 'ArrowRight' || event.key === 'ArrowUp' || 
        event.key === 'ArrowDown' || event.ctrlKey || event.metaKey) {
        return true;
    }
    
    // Permitir punto decimal solo una vez
    if (event.key === '.' || event.key === ',') {
        if (value.includes('.') || value.includes(',')) {
            event.preventDefault();
            showFieldError(input, 'Solo se permite un punto decimal');
            return false;
        }
        return true;
    }
    
    // Solo permitir números
    if (!/[0-9]/.test(event.key)) {
        event.preventDefault();
        showFieldError(input, 'Solo se permiten números');
        return false;
    }
    
    clearFieldError(input);
    return true;
}

// Validación para campos de precio (números con decimales)
function validatePriceInput(event) {
    const input = event.target;
    const value = input.value;
    
    // Permitir teclas de control
    if (event.key === 'Backspace' || event.key === 'Delete' || event.key === 'Tab' || 
        event.key === 'ArrowLeft' || event.key === 'ArrowRight' || event.key === 'ArrowUp' || 
        event.key === 'ArrowDown' || event.ctrlKey || event.metaKey) {
        return true;
    }
    
    // Permitir punto decimal
    if (event.key === '.' || event.key === ',') {
        if (value.includes('.') || value.includes(',')) {
            event.preventDefault();
            showFieldError(input, 'Solo se permite un punto decimal');
            return false;
        }
        return true;
    }
    
    // Solo permitir números
    if (!/[0-9]/.test(event.key)) {
        event.preventDefault();
        showFieldError(input, 'Solo se permiten números y punto decimal');
        return false;
    }
    
    clearFieldError(input);
    return true;
}

// Validación para campos de texto (sin caracteres especiales peligrosos)
function validateTextInput(event) {
    const input = event.target;
    const value = input.value;
    
    // Permitir teclas de control
    if (event.key === 'Backspace' || event.key === 'Delete' || event.key === 'Tab' || 
        event.key === 'ArrowLeft' || event.key === 'ArrowRight' || event.key === 'ArrowUp' || 
        event.key === 'ArrowDown' || event.ctrlKey || event.metaKey) {
        return true;
    }
    
    // Permitir letras, números, espacios y algunos caracteres especiales
    if (!/^[a-zA-ZáéíóúÁÉÍÓÚñÑ0-9\s\-_.,;:()]+$/.test(event.key)) {
        event.preventDefault();
        showFieldError(input, 'Caracteres especiales no permitidos');
        return false;
    }
    
    clearFieldError(input);
    return true;
}

// Validación para código de barras (solo números y letras)
function validateBarcodeInput(event) {
    const input = event.target;
    
    // Permitir teclas de control
    if (event.key === 'Backspace' || event.key === 'Delete' || event.key === 'Tab' || 
        event.key === 'ArrowLeft' || event.key === 'ArrowRight' || event.key === 'ArrowUp' || 
        event.key === 'ArrowDown' || event.ctrlKey || event.metaKey) {
        return true;
    }
    
    // Solo permitir números y letras
    if (!/[a-zA-Z0-9]/.test(event.key)) {
        event.preventDefault();
        showFieldError(input, 'Solo se permiten números y letras');
        return false;
    }
    
    clearFieldError(input);
    return true;
}

// Validación en tiempo real del valor
function validateFieldValue(input) {
    const value = input.value.trim();
    const fieldType = input.getAttribute('data-validation-type') || input.type;
    
    // Validar campo requerido
    if (input.hasAttribute('required') && !value) {
        showFieldError(input, 'Este campo es obligatorio');
        return false;
    }
    
    // Validaciones según tipo
    switch(fieldType) {
        case 'number':
        case 'price':
            if (value && (isNaN(value) || parseFloat(value) < 0)) {
                showFieldError(input, 'Debe ser un número válido mayor o igual a 0');
                return false;
            }
            if (input.hasAttribute('min') && parseFloat(value) < parseFloat(input.getAttribute('min'))) {
                showFieldError(input, `El valor mínimo es ${input.getAttribute('min')}`);
                return false;
            }
            if (input.hasAttribute('max') && parseFloat(value) > parseFloat(input.getAttribute('max'))) {
                showFieldError(input, `El valor máximo es ${input.getAttribute('max')}`);
                return false;
            }
            break;
            
        case 'text':
        case 'name':
            if (value.length < 2 && input.hasAttribute('required')) {
                showFieldError(input, 'El nombre debe tener al menos 2 caracteres');
                return false;
            }
            if (value.length > 100) {
                showFieldError(input, 'El nombre no puede exceder 100 caracteres');
                return false;
            }
            break;
            
        case 'barcode':
            if (value && value.length > 50) {
                showFieldError(input, 'El código de barras no puede exceder 50 caracteres');
                return false;
            }
            break;
            
        case 'description':
            if (value.length > 500) {
                showFieldError(input, 'La descripción no puede exceder 500 caracteres');
                return false;
            }
            break;
    }
    
    clearFieldError(input);
    return true;
}

// Mostrar error en campo
function showFieldError(input, message) {
    // Remover clases de éxito
    input.classList.remove('is-valid');
    input.classList.add('is-invalid');
    
    // Remover mensaje de error anterior
    const existingError = input.parentElement.querySelector('.invalid-feedback');
    if (existingError) {
        existingError.remove();
    }
    
    // Crear mensaje de error
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback';
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    
    // Insertar después del input
    if (input.parentElement.classList.contains('input-group')) {
        input.parentElement.parentElement.appendChild(errorDiv);
    } else {
        input.parentElement.appendChild(errorDiv);
    }
}

// Limpiar error del campo
function clearFieldError(input) {
    input.classList.remove('is-invalid');
    input.classList.add('is-valid');
    
    const errorDiv = input.parentElement.querySelector('.invalid-feedback');
    if (errorDiv) {
        errorDiv.remove();
    }
}

// Validar formulario completo
function validateForm(form) {
    let isValid = true;
    const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
    
    inputs.forEach(input => {
        if (!validateFieldValue(input)) {
            isValid = false;
        }
    });
    
    // Validar campos opcionales también
    const allInputs = form.querySelectorAll('input, select, textarea');
    allInputs.forEach(input => {
        if (input.value && !validateFieldValue(input)) {
            isValid = false;
        }
    });
    
    return isValid;
}

// Inicializar validaciones en formulario
function initFormValidation(form) {
    // Agregar validaciones según tipo de campo
    const inputs = form.querySelectorAll('input, textarea');
    
    inputs.forEach(input => {
        const type = input.type;
        const name = input.name;
        
        // Asignar tipo de validación
        if (name === 'price' || (type === 'number' && name.includes('price'))) {
            input.setAttribute('data-validation-type', 'price');
            input.addEventListener('keypress', validatePriceInput);
            input.addEventListener('paste', function(e) {
                e.preventDefault();
                const pastedText = (e.clipboardData || window.clipboardData).getData('text');
                const numericValue = pastedText.replace(/[^0-9.]/g, '');
                if (numericValue) {
                    input.value = numericValue;
                    validateFieldValue(input);
                }
            });
        } else if (type === 'number') {
            input.setAttribute('data-validation-type', 'number');
            input.addEventListener('keypress', validateNumericInput);
            input.addEventListener('paste', function(e) {
                e.preventDefault();
                const pastedText = (e.clipboardData || window.clipboardData).getData('text');
                const numericValue = pastedText.replace(/[^0-9]/g, '');
                if (numericValue) {
                    input.value = numericValue;
                    validateFieldValue(input);
                }
            });
        } else if (name === 'barcode') {
            input.setAttribute('data-validation-type', 'barcode');
            input.addEventListener('keypress', validateBarcodeInput);
        } else if (name === 'name') {
            input.setAttribute('data-validation-type', 'name');
            input.addEventListener('keypress', validateTextInput);
            input.setAttribute('maxlength', '100');
        } else if (name === 'description') {
            input.setAttribute('data-validation-type', 'description');
            input.setAttribute('maxlength', '500');
        }
        
        // Validación al perder foco
        input.addEventListener('blur', function() {
            validateFieldValue(input);
        });
        
        // Validación al escribir
        input.addEventListener('input', function() {
            if (input.value) {
                validateFieldValue(input);
            } else {
                clearFieldError(input);
            }
        });
    });
    
    // Validar al enviar formulario
    form.addEventListener('submit', function(e) {
        if (!validateForm(form)) {
            e.preventDefault();
            e.stopPropagation();
            
            // Mostrar mensaje general
            const firstInvalid = form.querySelector('.is-invalid');
            if (firstInvalid) {
                firstInvalid.focus();
                firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            
            showFormError(form, 'Por favor, corrige los errores en el formulario');
            return false;
        }
        
        form.classList.add('was-validated');
    });
}

// Mostrar error general del formulario
function showFormError(form, message) {
    // Remover mensaje anterior
    const existingAlert = form.querySelector('.form-validation-alert');
    if (existingAlert) {
        existingAlert.remove();
    }
    
    // Crear alerta
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger form-validation-alert';
    alertDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle me-2"></i>
        ${message}
    `;
    
    // Insertar al inicio del formulario
    form.insertBefore(alertDiv, form.firstChild);
    
    // Auto-ocultar después de 5 segundos
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// Formatear precio mientras se escribe
function formatPrice(input) {
    let value = input.value.replace(/[^0-9.]/g, '');
    
    // Asegurar solo un punto decimal
    const parts = value.split('.');
    if (parts.length > 2) {
        value = parts[0] + '.' + parts.slice(1).join('');
    }
    
    // Limitar decimales a 2
    if (parts.length === 2 && parts[1].length > 2) {
        value = parts[0] + '.' + parts[1].substring(0, 2);
    }
    
    input.value = value;
}

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', function() {
    // Inicializar validaciones en todos los formularios
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        initFormValidation(form);
    });
    
    // Formatear precios en tiempo real
    const priceInputs = document.querySelectorAll('input[name="price"], input[data-validation-type="price"]');
    priceInputs.forEach(input => {
        input.addEventListener('input', function() {
            formatPrice(input);
        });
    });
});
