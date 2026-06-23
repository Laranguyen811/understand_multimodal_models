from enum import Enum


class ModelType(Enum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    SPEECH = "speech"
    VIDEO = "video"
    SPATIOTEMPORAL = "spatiotemporal"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    THREE_DIMENSION = "3d"
    TIME_SERIES = "time series"
    MULTIMODAL = "multimodal"
