is_simple_core = True

if is_simple_core:
    from dzero.core_simple import Variable
    from dzero.core_simple import Function
    from dzero.core_simple import using_config
    from dzero.core_simple import no_grad
    from dzero.core_simple import as_array
    from dzero.core_simple import as_variable
    from dzero.core_simple import setup_variable
else:
    from dzero.core import Variable
    from dzero.core_ import Function
    from dzero.core import using_config
    from dzero.core import no_grad
    from dzero.core import as_array
    from dzero.core import as_variable
    from dzero.core import setup_variable
    
setup_variable()
  
