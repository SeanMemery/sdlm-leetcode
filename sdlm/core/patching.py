import builtins

_original_str = builtins.str

from .tensor_string import TensorString


class TensorStringContext:
    """Context manager for temporary TensorString activation."""
    
    def __init__(self):
        self.original_str = _original_str
    
    def __enter__(self):
        """Activate TensorString for this context."""
        builtins.str = TensorString
        return self
    
    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb
    ):
        """Restore original string."""
        builtins.str = self.original_str


class DSPyPatcher:
    """
    Specifically patches DSPy modules while preserving others.
    """
    
    def __init__(self):
        self.is_active = False
        self.patched_classes = {}
    
    def patch_dspy_classes(self):
        """
        Patch specific DSPy classes to use TensorString.
        """
        try:
            import dspy
            
            # Patch dspy.Predict
            if hasattr(dspy, 'Predict'):
                self._patch_predict_class(dspy.Predict)
            
            # Patch dspy.ChainOfThought
            if hasattr(dspy, 'ChainOfThought'):
                self._patch_cot_class(dspy.ChainOfThought)
            
            # Patch dspy.LM base class
            if hasattr(dspy, 'LM'):
                self._patch_lm_class(dspy.LM)
            
            print("✅ DSPy classes patched successfully")
            
        except ImportError:
            print("❌ DSPy not available for patching")
    
    def _patch_predict_class(
        self,
        predict_class
    ):
        """
        Patch Predict class to handle TensorString inputs/outputs.
        """
        original_forward = predict_class.forward
        
        def tensor_aware_forward(self, **kwargs):
            # Convert string inputs to TensorString
            tensor_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, _original_str) and not isinstance(value, TensorString):
                    tensor_kwargs[key] = TensorString(value)
                else:
                    tensor_kwargs[key] = value
            
            # Call original forward
            result = original_forward(self, **tensor_kwargs)
            
            # Convert string outputs to TensorString
            if hasattr(result, '__dict__'):
                for attr_name, attr_value in result.__dict__.items():
                    if isinstance(attr_value, _original_str) and not isinstance(attr_value, TensorString):
                        setattr(result, attr_name, TensorString(attr_value))
            
            return result
        
        predict_class.forward = tensor_aware_forward
        self.patched_classes['Predict'] = (predict_class, 'forward', original_forward)
    
    def _patch_cot_class(
        self,
        cot_class
    ):
        """
        Patch ChainOfThought class.
        """
        # Similar to Predict patching
        original_forward = cot_class.forward
        
        def tensor_aware_cot_forward(self, **kwargs):
            # Convert inputs to TensorString
            tensor_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, _original_str) and not isinstance(value, TensorString):
                    tensor_kwargs[key] = TensorString(value)
                else:
                    tensor_kwargs[key] = value
            
            result = original_forward(self, **tensor_kwargs)
            
            # Convert outputs to TensorString
            if hasattr(result, '__dict__'):
                for attr_name, attr_value in result.__dict__.items():
                    if isinstance(attr_value, _original_str) and not isinstance(attr_value, TensorString):
                        setattr(result, attr_name, TensorString(attr_value))
            
            return result
        
        cot_class.forward = tensor_aware_cot_forward
        self.patched_classes['ChainOfThought'] = (cot_class, 'forward', original_forward)
    
    def _patch_lm_class(
        self,
        lm_class
    ):
        """
        Patch LM class to return TensorString from basic_request.
        """
        original_basic_request = lm_class.basic_request
        
        def tensor_aware_basic_request(self, prompt, **kwargs):
            # Ensure prompt is TensorString
            if isinstance(prompt, _original_str) and not isinstance(prompt, TensorString):
                prompt = TensorString(prompt)
            
            # Call original method
            result = original_basic_request(self, prompt, **kwargs)
            
            # Convert response list to TensorString
            if isinstance(result, list):
                tensor_result = []
                for item in result:
                    if isinstance(item, _original_str) and not isinstance(item, TensorString):
                        tensor_result.append(TensorString(item))
                    else:
                        tensor_result.append(item)
                return tensor_result
            
            return result
        
        lm_class.basic_request = tensor_aware_basic_request
        self.patched_classes['LM'] = (lm_class, 'basic_request', original_basic_request)
    
    def activate(self):
        """
        Activate DSPy-specific patching.
        """
        if self.is_active:
            return
        
        self.patch_dspy_classes()
        self.is_active = True
    
    def deactivate(self):
        """
        Restore original DSPy classes.
        """
        if not self.is_active:
            return
        
        # Restore all patched classes
        for class_name, (cls, method_name, original_method) in self.patched_classes.items():
            setattr(cls, method_name, original_method)
            print(f"Restored {class_name}.{method_name}")
        
        self.patched_classes.clear()
        self.is_active = False

