from pkgutil import iter_modules
from importlib import import_module
from types import ModuleType

from loguru import logger


def convert_snake_to_pascal(name: str) -> str:
    """snake_to_pascal -> SnakeToPascal"""
    return "".join(map(str.capitalize, name.split("_")))


def get_main_object_name(module: ModuleType) -> str:
    module_name = module.__name__.rsplit(".", 1)[-1]
    main_object_name = convert_snake_to_pascal(module_name)
    return main_object_name


def get_main_object(module: ModuleType, obj_type):
    """tries to find SomeObject in object src.some_object"""
    main_object_name = get_main_object_name(module)
    try:
        res = getattr(module, main_object_name)
        if not isinstance(res, obj_type) and not issubclass(res, obj_type):
            logger.error(f"Failed to load {module}.{main_object_name}: expected {obj_type} but found {type(res)}")
            return None
        return res
    except AttributeError as ex:
        logger.error(f"Failed to load {module}.{main_object_name}: {ex}")
        return None


def load_modules(package_name: str) -> list[ModuleType]:
    try:
        package = import_module(package_name)
    except ModuleNotFoundError:
        logger.error(f"Not found module: {package_name}")
        return []
    res = [import_module(module_name) for _, module_name, _ in iter_modules(package.__path__, package_name + ".")]
    return res


def load_main_objects(package_name: str, obj_type: type) -> dict[str, object]:
    modules = load_modules(package_name)
    main_objects = [get_main_object(m, obj_type) for m in modules]
    res = {obj.__name__: obj for obj in main_objects}
    return res
