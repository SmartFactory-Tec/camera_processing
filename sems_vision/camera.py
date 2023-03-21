from dataclasses import dataclass


@dataclass
class Camera:
    id: int
    name: str
    connection_string: str
    location_text: str
    location_id: int
