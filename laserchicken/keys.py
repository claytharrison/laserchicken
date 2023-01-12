# Name of point data section in point cloud structure
point = 'vertex'

# Name of the normalized height point attribute
normalized_height = 'normalized_height'

# Name of the intensity point attribute
intensity = 'intensity'

#
point_cloud = 'pointcloud'

#
provenance = 'log'

# Name of the amplitude point attribute
amplitude = 'Amplitude'

# Name of the pulse width attribute
pulse_width = 'Pulse width'

#name of the TEST bit fields attribute
bit_fields = 'bit_fields'

#names of classifications by ASPRS LAS specification 1.4

point_classes = {
    0: 'neverClassified', # created, never classified
    1: 'unclassified',
    2: 'ground',
    3: 'vegetation', # lowVegetation
    4: 'vegetation', # mediumVegetation
    5: 'vegetation', # highVegetation
    6: 'building',
    7: 'noise', # low point (noise)
    8: 'keyPoint', # model key-point (mass point)
    9: 'water',
    12: 'overlap', # Overlap Points
}
