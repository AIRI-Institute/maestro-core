from pydantic import Field

from mmar_mapi.models.base import Base


class TrackInfo(Base):
    track_id: str = Field(alias="TrackId")
    name: str = Field(alias="Name")
    domain_id: str = Field(alias="DomainId")


class DomainInfo(Base):
    domain_id: str = Field(alias="DomainId")
    name: str = Field(alias="Name")
