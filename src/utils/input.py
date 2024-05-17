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

# Auxiliar method to check if the label has an HFO event
def label_has_hfo_event(label):
    """
    This function checks if the label has an HFO event.
    @label (str): Label of the marker.
    """
    return "Ripple" in label or "Fast-Ripple" in label