import threading

class GoogleSheetLogger:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._active = False

    def activate(self): self._active = False
    def deactivate(self): self._active = False
    def shutdown(self): pass

    def log_message(self, sheet_name, row_data): pass
    def log_value(self, value): pass
    def log_error(self, message): pass
    def log_info(self, message): pass
    def log_order(self, order, pm_gross=0, pm_net=0): pass

gsheet_logger = GoogleSheetLogger()

class Arbiter:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self._active = False

    def activate(self): self._active = False
    def deactivate(self): self._active = False
    def shutdown(self): pass

    def get_param(self, owner, param, param_type=str, default_value=None):
        return default_value

arbiter = Arbiter()
