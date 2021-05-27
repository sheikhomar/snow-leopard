import dataclasses
import json

from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional

from pyArango.connection import Connection


@dataclass
class AttributeValue:
    type_name: str
    value: object


@dataclass
class MultiValue:
    attribute_id: str
    values: List[AttributeValue] = field(default_factory=list)


@dataclass
class ProductImage:
    type: str
    url: str
    width: int
    height: int
    size_bytes: int


@dataclass
class ProductEAN:
    variant_id: Optional[str]
    ean: str


@dataclass
class ProductFeature:
    id: str
    category_id: str
    group_id: str
    measure_id: str
    name: str
    value: str
    presentation_value: str
    is_translated: bool
    is_mandatory: bool
    is_searchable: bool


@dataclass
class Product:
    id: int
    supplier_id: int
    category_id: int
    created_at: datetime
    updated_at: datetime
    is_limited: bool
    on_market: bool
    quality: str
    model_name: str
    released_on: Optional[datetime] = field(default=None)
    end_of_life_on: Optional[datetime] = field(default=None)
    title: str = field(default='')
    category_name: str = field(default='')
    description_short: str = field(default='')
    description_middle: str = field(default='')
    description_long: str = field(default='')
    warranty: str = field(default='')
    url_details: str = field(default='')
    url_manual: str = field(default='')
    url_pdf: str = field(default='')

    images: List[ProductImage] = field(default_factory=list)
    country_markets: List[str] = field(default_factory=list)
    ean: List[ProductEAN] = field(default_factory=list)
    features: List[ProductFeature] = field(default_factory=list)


class Repository:
    def __init__(self, db_auth_path: str) -> None:
        with open(db_auth_path, "r", encoding="utf8") as fp:
            db_settings = json.load(fp)
        conn = Connection(
            username=db_settings["username"],
            password=db_settings["password"], 
        )
        db_name = db_settings["database"]
        if conn.hasDatabase(db_name):
            self._db = conn[db_name]
        else:
            self._db = conn.createDatabase(db_name)
        self._conn = conn
        self._products = None

    @property
    def products(self):
        if self._products is None:
            collection_name = 'products'
            if self._db.hasCollection(collection_name):
                self._products =  self._db[collection_name]
            else:
                self._products =  self._db.createCollection(name=collection_name)
        return self._products

    def create_product(self, product: Product) -> None:
        document = self.products.createDocument(dataclasses.asdict(product))
        document._key = str(product.id)
        document.save()

    def product_exists(self, product_id: str) -> bool:
        return (product_id in self.products)


