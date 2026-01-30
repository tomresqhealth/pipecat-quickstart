# videos.py

# A dictionary mapping keywords (what the LLM calls) to metadata (what the User sees + descriptions)
VIDEO_DATABASE = {
    "skiing": {
        "url": "https://nutum.ai/wp-content/uploads/2026/01/tom_skiing_final.mp4",
        "description": "A video of Tom Elliott skiing at Mount Sunapee in 2024. Use this when the user asks about his hobbies, sports, or skiing."
    }
}

def get_video_url(keyword: str) -> str:
    """
    Retrieves the URL for a given keyword. 
    Defaults to an empty string if the key doesn't exist.
    """
    data = VIDEO_DATABASE.get(keyword.lower())
    if data:
        return data["url"]
    return ""

def get_video_descriptions() -> str:
    """
    Generates a string description of all available videos for the System Prompt.
    Format:
    - keyword: Description
    """
    lines = []
    for key, data in VIDEO_DATABASE.items():
        lines.append(f"- '{key}': {data['description']}")
    return "\n".join(lines)