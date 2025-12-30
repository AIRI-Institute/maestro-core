from collections.abc import Sequence
from typing import Any

from dishka.dependency_source import (
    CompositeDependencySource,
    Factory,
)
from dishka.entities.factory_type import FactoryType
from dishka.entities.key import (
    hint_to_dependency_key,
)
from dishka.entities.scope import BaseScope
from dishka.provider.make_factory import _provide
from dishka.provider.unpack_provides import unpack_factory


# hotfix for `dishka` library: will be eliminated in the future
def provide_all_and_memoize(
    *,
    provides: Sequence[Any],
    provides_result: Any,
    scope: BaseScope | None = None,
    cache: bool = True,
    is_in_class: bool = True,
    recursive: bool = False,
    override: bool = False,
) -> CompositeDependencySource:
    composite = CompositeDependencySource(None)

    # 1. Register each individual provider
    individual_factories: list[Factory] = []
    for single_provides in provides:
        src = _provide(
            source=single_provides,
            provides=None,
            scope=scope,
            cache=cache,
            is_in_class=is_in_class,
            recursive=recursive,
            override=override,
        )
        composite.dependency_sources.extend(src.dependency_sources)
        for ds in src.dependency_sources:
            if isinstance(ds, Factory):
                individual_factories.append(ds)

    # 2. Build memoizing result factory
    # Its dependencies are exactly the provides list
    dependency_keys = [hint_to_dependency_key(p) for p in provides]

    def memoized_factory(*items):
        # return same container type as user passed (tuple/list/etc.)
        if isinstance(provides_result, tuple):
            return tuple(items)
        return list(items)

    memo_factory = Factory(
        dependencies=dependency_keys,
        kw_dependencies={},
        type_=FactoryType.FACTORY,
        source=memoized_factory,
        scope=scope,
        provides=hint_to_dependency_key(provides_result),
        is_to_bind=False,
        cache=cache,
        override=override,
    )

    composite.dependency_sources.extend(unpack_factory(memo_factory))
    return composite
