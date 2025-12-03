from pydantic import BaseModel


class ResourcesModel(BaseModel):
    def __repr__(self):
        class_name = str(type(self).__name__)
        fields = self.__dict__.keys()
        return f"{class_name}({', '.join(fields)})"
