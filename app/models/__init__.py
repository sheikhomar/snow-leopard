import dataclasses
import json

from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional

from pyArango.connection import Connection


@dataclass
class Product:
    id: int
    supplier_id: int
    category_id: int
    created_at: datetime
    updated_at: datetime
    is_limited: bool
    on_market: bool
    image_url: str
    image_width: int
    image_height: int
    quality: str
    model_name: str


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


