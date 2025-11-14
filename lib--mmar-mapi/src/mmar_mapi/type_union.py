from typing import Any, Union, Annotated, get_args, get_origin, Literal

from pydantic import Discriminator, BaseModel, Tag


class _TypeUnionMeta(type):
    def __getitem__(cls, types):
        if not isinstance(types, tuple):
            types = (types,)

        for tp in types:
            if not isinstance(tp, type) or not issubclass(tp, BaseModel):
                raise ValueError(f"Type {tp} must derived from BaseModel")

        # Create tagged union
        tagged_types = []
        type_map = {}
        type_names = set()
        for t in types:
            type_annot = t.__annotations__.get("type")
            if not type_annot or get_origin(type_annot) != Literal:
                raise ValueError(f"Type {t} must have a 'type' Literal[..] field for discrimination")
            type_name = get_args(type_annot)[0]
            type_map[type_name] = t
            tagged_types.append(Annotated[t, Tag(type_name)])
            type_names.add(type_name)

        # Create the union type
        union_type = Union[tuple(tagged_types)]

        # Create discriminator function
        def type_discriminator(v: Any) -> str | None:
            if isinstance(v, types):
                return v.type
            if isinstance(v, dict):
                tp = v.get("type")
                return tp if tp in type_names else None
            return None

        # Add discriminator
        return Annotated[union_type, Discriminator(type_discriminator)]


class TypeUnion(metaclass=_TypeUnionMeta):
    """
    Wrapper around Union which derive type from field 'type' for effective deserialization via TypeAdapter.

    Usage example:
    ```
    class BaseFruit(BaseModel):
        type: str

    class Apple(BaseFruit):
        type: Literal['apple'] = 'apple'

    class Orange(BaseFruit):
        type: Literal['orange'] = 'orange'

    Fruit = TypeUnion[Apple, Orange]
    ```
    """

    pass
