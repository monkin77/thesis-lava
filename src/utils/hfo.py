# Create enum for the different types of markers
class MarkerType:
    """Enum for the different types of markers"""
    RIPPLE = 1
    FAST_RIPPLE = 2
    BOTH = 3
    OTHER = 4

RIPPLE_BAND_FILENAME = "ripple"
FR_BAND_FILENAME = "fr"
BOTH_BAND_FILENAME = "hfo"

def band_to_file_name(band: MarkerType):
    """
    This function returns the file name for the band.
    @band (MarkerType): The band to get the file name for.
    """
    if band == MarkerType.RIPPLE:
        return RIPPLE_BAND_FILENAME
    elif band == MarkerType.FAST_RIPPLE:
        return FR_BAND_FILENAME
    elif band == MarkerType.BOTH:
        return BOTH_BAND_FILENAME
    else:
        return "unknown"
    
class BaselineAlgorithm:
    """Enum for the different types of baseline algorithms"""
    Q1 = "q1"
    MEAN = "mean"
    MEDIAN = "median"
    Q3 = "q3"
    SIXTY_PERC = "60perc"
    SIXTY_FIVE_PERC = "65perc"
    SEVENTY_PERC = "70perc"
    EIGHTY_PERC = "80perc"
    NINETY_PERC = "90perc"

# HFO Detection Offsets [MIN_OFFSET, MAX_OFFSET, MEAN_OFFSET, TOLERANCE_OFFSET]
RIPPLE_DETECTION_OFFSET = [18, 57, 37, 20]
FR_DETECTION_OFFSET = [9, 18, 13, 5]
BOTH_DETECTION_OFFSET = [9, 57, 33, 24]

def band_to_gt_max_offset(band: MarkerType):
    """
    This function returns the Maximum time the SNN can take to detect the event.
    If the band is unknown, it returns 0.
    @band (MarkerType): The band to get the confidence window for.
    """
    if band == MarkerType.RIPPLE:
        return RIPPLE_DETECTION_OFFSET[1]
    elif band == MarkerType.FAST_RIPPLE:
        return FR_DETECTION_OFFSET[1]
    elif band == MarkerType.BOTH:
        return BOTH_DETECTION_OFFSET[1]
    else:
        raise ValueError("Unknown band type on band_to_confidence_window()")