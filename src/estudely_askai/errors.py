class AppError(Exception):
    exit_code = 1

    def __init__(self, message: str):
        super().__init__(message)


class ConnectionError(AppError):
    exit_code = 2


class OllamaAPIError(AppError):
    exit_code = 3


class ConfigError(AppError):
    exit_code = 4
