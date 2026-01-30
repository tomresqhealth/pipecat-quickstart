# images.py

# A dictionary mapping keywords (what the LLM calls) to metadata (what the User sees + descriptions)
IMAGE_DATABASE = {
    "birthday": {
        "url": "https://nutum.ai/wp-content/uploads/2026/01/tom_birthday.jpg",
        "description": "A picture of Tom Elliott celebrating his 52nd birthday at his house. He is smiling and wearing a green shirt. Use this when the user asks about his age, birthday, or personal milestones."
    },
    "turkey_vacation": {
        "url": "https://nutum.ai/wp-content/uploads/2026/01/Tom-in-Turkey-Volney-scaled.jpeg",
        "description": "A picture of Tom Elliott visiting an ancient site in Turkey with his nephew Volney. Tom is wearing a black shirt and glasses. Use this when the user asks about his travels, vacations, or family trips."
    },
    "fenway_park": {
        "url": "https://nutum.ai/wp-content/uploads/2026/01/Tom-Fenway-Park-Bear-scaled.jpeg",
        "description": "A picture of Tom Elliott and his son Bear at a Red Sox game at Fenway Park in Boston (Sept 2025). Tom is wearing a pink salmon sweater. Use this when the user asks about his hobbies, sports, or time spent with his son."
    },
    "headshot": {
        "url": "https://nutum.ai/wp-content/uploads/2025/03/tom-headshot-2024.jpeg",
        "description": "A professional headshot of Tom Elliott wearing a blue vest. Use this when the user asks who Tom is, what he looks like, or for a professional photo."
    }
}

def get_image_url(keyword: str) -> str:
    """
    Retrieves the URL for a given keyword. 
    Defaults to a placeholder if the key doesn't exist.
    """
    # Normalize key to lower case to prevent mismatch errors
    key = keyword.lower().strip()
    data = IMAGE_DATABASE.get(key)
    
    if data:
        return data["url"]
    
    # Fallback image if the LLM hallucinates a keyword
    return "https://placehold.co/600x400?text=Image+Not+Found"

def get_image_descriptions() -> str:
    """
    Generates a string description of all available images for the System Prompt.
    Format:
    - keyword: Description
    """
    lines = []
    for key, data in IMAGE_DATABASE.items():
        lines.append(f"- '{key}': {data['description']}")
    return "\n".join(lines)