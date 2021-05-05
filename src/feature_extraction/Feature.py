class Feature:
    def __init__(self, value: float, name: str, extraction_properties: dict = None):
        self.value = value
        self.name = name
        self.extraction_properties = extraction_properties

    def __float__(self):
        return self.value

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def get_extraction_properties(self):
        return self.extraction_properties
