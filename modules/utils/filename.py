import re


def safe_filename(name: str) -> str:
    invalid_filename_chars = r'[<>:"/\\|?*\x00-\x1f]'
    max_filename_length = 200
    safe_name = re.sub(invalid_filename_chars, "_", name)

    if len(safe_name) > max_filename_length:
        file_extension = safe_name.split(".")[-1]
        if len(file_extension) + 1 < max_filename_length:
            truncated_name = safe_name[:max_filename_length - len(file_extension) - 1]
            safe_name = truncated_name + "." + file_extension
        else:
            safe_name = safe_name[:max_filename_length]

    return safe_name
