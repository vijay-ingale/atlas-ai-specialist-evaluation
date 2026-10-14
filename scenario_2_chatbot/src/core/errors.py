class ApplicationError(Exception):
    status_code = 500
    code = "application_error"

class InvalidInputError(ApplicationError):
    status_code = 400
    code = "invalid_input"

class IndexNotReadyError(ApplicationError):
    status_code = 409
    code = "index_not_ready"

class ConfigurationError(ApplicationError):
    status_code = 503
    code = "configuration_error"

class ProviderError(ApplicationError):
    status_code = 502
    code = "provider_error"

class StorageError(ApplicationError):
    status_code = 503
    code = "storage_error"
