import pandas as pd

def read_spike_events(file_path: str):
    """Reads the spike events from the input file and returns them as a numpy array
    TODO: Could try to use a structured array instead?

    Args:
        file_path (str): name of the file containing the spike events
    """
    spike_events = []

    try:
        # Read the spike events from the file
        df = pd.read_csv(file_path, header='infer')
        # print(df.head())

        # Detect errors
        if df.empty:
            raise Exception("The input file is empty")

        # Convert the scientific notation values to integers if any exist
        df = df.applymap(lambda x: int(float(x)) if (isinstance(x, str) and 'e' in x) else x)

        # Convert the dataframe to a numpy array. Each row is a spike event. The first column is the time and the second column is the channel idx
        spike_events = df.to_numpy()

        return spike_events
    except Exception as e:
        print("Unable to read the input file: ", file_path, " error:", e)

    return spike_events

# Create enum for the different types of markers
class MarkerType:
    """Enum for the different types of markers"""
    RIPPLE = 1
    FAST_RIPPLE = 2
    BOTH = 3
    OTHER = 4

class BaselineAlgorithm:
    """Enum for the different types of baseline algorithms"""
    Q1 = "q1"
    MEAN = "mean"
    MEDIAN = "median"
    Q3 = "q3"
    SIXTY_PERC = "60perc"
    SIXTY_FIVE_PERC = "65perc"
    SEVENTY_PERC = "70perc"


# Auxiliar method to check if the label has an HFO event
def label_has_hfo_event(label):
    """
    This function checks if the label has an HFO event.
    @label (str): Label of the marker.
    """
    return "Ripple" in label or "Fast-Ripple" in label

# Auxiliar method to check if the label has a Ripple
def label_has_ripple(label):
    """
    This function checks if the label has a Ripple.
    @label (str): Label of the marker.
    """
    # Ensure the "Ripple" is not part of the "Fast-Ripple
    if "Fast-Ripple" in label:
        # Remove the "Fast-Ripple" from the label
        label = label.replace("Fast-Ripple", "")
    
    return "Ripple" in label

# Auxiliar method to check if the label has a Fast Ripple
def label_has_fast_ripple(label):
    """
    This function checks if the label has a Fast Ripple.
    @label (str): Label of the marker.
    """
    return "Fast-Ripple" in label

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
    
def band_to_confidence_window(band: MarkerType):
    """
    This function returns the confidence window for the band.
    If the band is unknown, it returns 0.
    @band (MarkerType): The band to get the confidence window for.
    """
    if band == MarkerType.RIPPLE:
        return RIPPLE_CONFIDENCE_WINDOW
    elif band == MarkerType.FAST_RIPPLE:
        return FR_CONFIDENCE_WINDOW
    elif band == MarkerType.BOTH:
        return BOTH_CONFIDENCE_WINDOW
    else:
        raise ValueError("Unknown band type on band_to_confidence_window()")
    

SAMPLING_RATE = 2048    # 2048 Hz
INPUT_DURATION_S = 120    # 120 seconds
INPUT_DURATION = INPUT_DURATION_S * (10**3)    # 120000 ms or 120 seconds
NUM_SAMPLES = SAMPLING_RATE * INPUT_DURATION_S  # 2048 samples per second for 120 seconds

X_STEP = 1/SAMPLING_RATE * (10**3)  # 0.48828125 ms

RIPPLE_CONFIDENCE_WINDOW = 120  # Let's give a 120ms window after the Ripple to consider as part of the event
FR_CONFIDENCE_WINDOW = 60  # Let's give a 60ms window after the Fast Ripple to consider as part of the event    (TODO: Could be changed)
BOTH_CONFIDENCE_WINDOW = 120  # Let's give a 120ms window after the HFO event (Ripple or Fast Ripple) insertion to consider as part of the event

class ModelDistStrategy:
    """Enum for the Selected Distribution Strategy"""
    IQR = 0
    MEAN_AND_STD = 1
    LOG_NORMAL = 2  # https://planetcalc.com/7264/ and https://www.omnicalculator.com/statistics/lognormal-distribution
                    