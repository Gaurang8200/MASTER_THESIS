# Definition der Zonenkoordinaten
ZONE_COORDINATES = {
    "Zone_1": {"x": 0.366, "y": -0.146, "z": 0.067},  
    "Zone_2": {"x": 0.366, "y": -0.008, "z": 0.048},  
    "Zone_3": {"x": 0.366, "y": 0.121, "z": 0.040},
    "Zone_4": {"x": 0.500, "y": -0.146, "z": 0.067},
    "Zone_5": {"x": 0.500, "y": 0.121, "z": 0.040},
    # Aliase für verschiedene Schreibweisen
    "zone_1": {"x": 0.366, "y": -0.146, "z": 0.067},
    "zone_2": {"x": 0.366, "y": -0.008, "z": 0.048},
    "zone_3": {"x": 0.366, "y": 0.121, "z": 0.040}
}

# Objektspezifische Abstellhöhen
OBJECT_PLACE_HEIGHTS = {
    0: 0.067,  # Cylinder
    1: 0.048,  # Box  
    2: 0.040   # Marker
}

def get_zone_coordinates(zone_name: str) -> dict:
    """
    Gibt die Koordinaten für eine bestimmte Zone zurück.
    
    Args:
        zone_name: Name der Zone (z.B. "Zone_1")
        
    Returns:
        Dictionary mit x, y, z Koordinaten
    """
    return ZONE_COORDINATES.get(zone_name, None)

def get_object_place_height(object_class: int) -> float:
    """
    Gibt die objektspezifische Abstellhöhe zurück.
    
    Args:
        object_class: Objektklasse (0=Cylinder, 1=Box, 2=Marker)
        
    Returns:
        Z-Koordinate für das Abstellen des Objekts
    """
    return OBJECT_PLACE_HEIGHTS.get(object_class, 0.067)  # Default: Cylinder-Höhe
