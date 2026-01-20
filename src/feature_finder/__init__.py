import os
import warnings


def custom_formatwarning(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"


# Overwrite warnings functionality
warnings.formatwarning = custom_formatwarning
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Define max CPU cores of current machine
MAX_CORES = min(int(os.cpu_count() / 6), 4)
