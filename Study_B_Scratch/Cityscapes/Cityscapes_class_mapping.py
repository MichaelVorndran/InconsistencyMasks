
# RGB

COLOR_TO_CLASS_MAPPING_CITYSCAPES_ORG = {
    (0, 0, 0): 0,    # Unlabeled
    (0, 0, 128): 1,  # Ego vehicle
    (0, 128, 0): 2,  # Rectification border
    (0, 128, 128): 3,  # Out of roi
    (128, 0, 0): 4,  # Static
    (128, 0, 128): 5,  # Dynamic
    (128, 128, 0): 6,  # Ground
    (128, 128, 128): 7,  # Road
    (0, 0, 64): 8,  # Sidewalk
    (0, 0, 192): 9,  # Parking
    (0, 128, 64): 10,  # Rail track
    (0, 128, 192): 11,  # Building
    (128, 0, 64): 12,  # Wall
    (128, 0, 192): 13,  # Fence
    (128, 128, 64): 14,  # Guard rail
    (128, 128, 192): 15,  # Bridge
    (0, 64, 0): 16,  # Tunnel
    (0, 64, 128): 17,  # Pole
    (0, 192, 0): 18,  # Polegroup
    (0, 192, 128): 19,  # Traffic light
    (128, 64, 0): 20,  # Traffic sign
    (128, 64, 128): 21,  # Vegetation
    (128, 192, 0): 22,  # Terrain
    (128, 192, 128): 23,  # Sky
    (0, 64, 64): 24,  # Person
    (0, 64, 192): 25,  # Rider
    (0, 192, 64): 26,  # Car
    (0, 192, 192): 27,  # Truck
    (128, 64, 64): 28,  # Bus
    (128, 64, 192): 29,  # Caravan
    (128, 192, 64): 30,  # Trailer
    (128, 192, 192): 31,  # Train
    (64, 0, 0): 32,  # Motorcycle
    (64, 0, 128): 33,  # Bicycle
    (192, 192, 192): -1  # License plate
}


COLOR_TO_CLASS_MAPPING_CITYSCAPES = {
    (0, 0, 0): 0,    # IM
    (0, 0, 128): 1,  # Unlabeled
    (0, 128, 0): 2,  # Ego vehicle
    (0, 128, 128): 3,  # Rectification border
    (128, 0, 0): 4,  # Out of roi
    (128, 0, 128): 5,  # Static
    (128, 128, 0): 6,  # Dynamic
    (128, 128, 128): 7,  # Ground
    (0, 0, 64): 8,  # Road
    (0, 0, 192): 9,  # Sidewalk
    (0, 128, 64): 10,  # Parking
    (0, 128, 192): 11,  # Rail track
    (128, 0, 64): 12,  # Building
    (128, 0, 192): 13,  # Wall
    (128, 128, 64): 14,  # Fence
    (128, 128, 192): 15,  # Guard rail
    (0, 64, 0): 16,  # Bridge
    (0, 64, 128): 17,  # Tunnel
    (0, 192, 0): 18,  # Pole
    (0, 192, 128): 19,  # Polegroup
    (128, 64, 0): 20,  # Traffic light
    (128, 64, 128): 21,  # Traffic sign
    (128, 192, 0): 22,  # Vegetation
    (128, 192, 128): 23,  # Terrain
    (0, 64, 64): 24,  # Sky
    (0, 64, 192): 25,  # Person
    (0, 192, 64): 26,  # Rider
    (0, 192, 192): 27,  # Car
    (128, 64, 64): 28,  # Truck
    (128, 64, 192): 29,  # Bus
    (128, 192, 64): 30,  # Caravan
    (128, 192, 192): 31,  # Trailer
    (64, 0, 0): 32,  # Train
    (64, 0, 128): 33,  # Motorcycle
    (64, 128, 0): 34,  # Bicycle
    (192, 192, 192): -1  # License plate
}


CLASS_DESCRIPTION = {
    0: 'IM',
    1: 'Unlabeled',
    2: 'Ego vehicle',
    3: 'Rectification border',
    4: 'Out of roi',
    5: 'Static',
    6: 'Dynamic',
    7: 'Ground',
    8: 'Road',
    9: 'Sidewalk',
    10: 'Parking',
    11: 'Rail track',
    12: 'Building',
    13: 'Wall',
    14: 'Fence',
    15: 'Guard rail',
    16: 'Bridge',
    17: 'Tunnel',
    18: 'Pole',
    19: 'Polegroup',
    20: 'Traffic light',
    21: 'Traffic sign',
    22: 'Vegetation',
    23: 'Terrain',
    24: 'Sky',
    25: 'Person',
    26: 'Rider',
    27: 'Car',
    28: 'Truck',
    29: 'Bus',
    30: 'Caravan',
    31: 'Trailer',
    32: 'Train',
    33: 'Motorcycle',
    34: 'Bicycle',
    -1: 'License plate',
}
