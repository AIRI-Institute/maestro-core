ResourceId = str


class TextExtractorAPI:
    def extract(self, *, resource_id: ResourceId) -> ResourceId:
        """returns file with text"""
        raise NotImplementedError
