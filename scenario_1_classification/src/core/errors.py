class ApplicationError(Exception):
    status_code = 500
    code = "application_error"

class InvalidInputError(ApplicationError):
    status_code = 400
    code = "invalid_input"

class ModelNotReadyError(ApplicationError):
    status_code = 409
    code = "model_not_ready"

class StorageError(ApplicationError):
    status_code = 503
    code = "storage_error"
