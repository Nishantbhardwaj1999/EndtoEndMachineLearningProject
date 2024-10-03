import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logger import logging

# Ensure the src path is added to the system path


def error_message_detail(error, error_detail: sys):
    """
    Constructs a detailed error message.
    
    Parameters:
    - error: The original error that was raised.
    - error_detail: The sys module, used to extract traceback information.
    
    Returns:
    - A formatted string with details about the error.
    """
    _, _, exc_tb = error_detail.exc_info()  # Get traceback information
    file_name = exc_tb.tb_frame.f_code.co_filename  # Name of the script where the error occurred
    error_message = (
        "Error occurred in Python script name [{0}] line number [{1}] error message [{2}]"
    ).format(file_name, exc_tb.tb_lineno, str(error))  # Format the error message

    return error_message

class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        """
        Initializes the custom exception with a formatted error message.
        
        Parameters:
        - error_message: The original error message.
        - error_detail: The sys module, used to extract traceback information.
        """
        super().__init__(error_message)  # Initialize the base Exception class
        self.error_message = error_message_detail(error_message, error_detail=error_detail)  # Format the error message
    
    def __str__(self):
        return self.error_message  # Return the formatted error message when str() is called

# if __name__ == "__main__":
#     try:
#         a = 1 / 10  # This will raise a ZeroDivisionError
#     except Exception as e:
#         logging.info("Logging is started.")  # Log the start of logging
#         raise CustomException(e, sys)  # Raise the custom exception with original error and sys module
