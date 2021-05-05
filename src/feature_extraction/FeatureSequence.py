class FeatureSequence:
    """
    Basic entity to store a sequence of feature values, their type and extraction properties.
    It use only primitive types to occupy the minimum space to be serialized efficiently.
    """
    def __init__(self, values_sequence: [], feature_name: str, extraction_properties: dict = None):
        self.values = values_sequence
        self.name = feature_name
        self.extraction_properties = extraction_properties

    def __float__(self):
        return self.value

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def get_extraction_properties(self):
        return self.extraction_properties
