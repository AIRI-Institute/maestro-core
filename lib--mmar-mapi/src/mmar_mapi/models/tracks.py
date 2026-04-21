from mmar_mapi.models.base import Base


class TrackInfo(Base):
    track_id: str
    name: str
    domain_id: str


class DomainInfo(Base):
    domain_id: str
    name: str
