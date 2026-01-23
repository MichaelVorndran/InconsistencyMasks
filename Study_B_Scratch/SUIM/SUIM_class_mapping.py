
# RGB

COLOR_TO_CLASS_MAPPING_SUIM = {
    (211, 211, 211): 0,   # IM
    (0, 0, 0): 1,    # Background (waterbody)
    (0, 0, 255): 2,  # Human divers
    (0, 255, 0): 3,  # Aquatic plants and sea-grass
    (0, 255, 255): 4,  # Wrecks and ruins
    (255, 0, 0): 5,  # Robots (AUVs/ROVs/instruments)
    (255, 0, 255): 6,  # Reefs and invertebrates
    (255, 255, 0): 7,  # Fish and vertebrates
    (255, 255, 255): 8  # Sea-floor and rocks
}

COLOR_TO_CLASS_MAPPING_SUIM_ORG = {
    (0, 0, 0): 0,    # Background (waterbody)
    (0, 0, 255): 1,  # Human divers
    (0, 255, 0): 2,  # Aquatic plants and sea-grass
    (0, 255, 255): 3,  # Wrecks and ruins
    (255, 0, 0): 4,  # Robots (AUVs/ROVs/instruments)
    (255, 0, 255): 5,  # Reefs and invertebrates
    (255, 255, 0): 6,  # Fish and vertebrates
    (255, 255, 255): 7  # Sea-floor and rocks
}


CLASS_DESCRIPTION = {
    0: 'IM',
    1: 'Background (waterbody)',
    2: 'Human divers', 
    3: 'Aquatic plants and sea-grass',
    4: 'Wrecks and ruins',
    5: 'Robots (AUVs/ROVs/instruments)',
    6: 'Reefs and invertebrates',
    7: 'Fish and vertebrates',
    8: 'Sea-floor and rocks'
}
