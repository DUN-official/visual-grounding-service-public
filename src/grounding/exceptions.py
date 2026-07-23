class GroundingError(RuntimeError):
    pass

class BackendNotReadyError(GroundingError):
    pass

class ModelProvisioningError(GroundingError):
    pass

class ImageInputError(GroundingError):
    pass

class RemoteBackendError(GroundingError):
    pass
