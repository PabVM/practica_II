def zfill(string: str, length: int) -> str:
    """Adds zeroes at the begginning of a string 
    until it completes the desired length."""
    return '0' * (length - len(string)) + string