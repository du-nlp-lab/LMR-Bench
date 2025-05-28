def file_to_string(file_path):
    """
    Reads the contents of a file and returns it as a string.

    Parameters:
        file_path (str): The path to the file to read.

    Returns:
        str: The contents of the file as a single string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: The file at {file_path} was not found."
    except Exception as e:
        return f"An error occurred: {e}"
