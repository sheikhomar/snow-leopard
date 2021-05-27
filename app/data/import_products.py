from typing import List
from datetime import datetime
from pathlib import Path

import click
from lxml import etree
from app.models import *

def parse_product_info(file_path: Path) -> List[MultiValue]:
    pass


def extract_product_info(xml_file_element) -> Product:
    attrs = dict(xml_file_element.items())

    product = Product(
        id = int(attrs["Product_ID"]),
        supplier_id = int(attrs["Supplier_id"]),
        category_id = int(attrs["Catid"]),
        created_at = datetime.strptime(attrs["Date_Added"], "%Y%m%d%H%M%S"),
        updated_at = datetime.strptime(attrs["Updated"], "%Y%m%d%H%M%S"),
        is_limited = attrs["Limited"] == "Yes",
        on_market = attrs["On_Market"] == "1",
        image_url = attrs["HighPic"],
        image_width = int(attrs["HighPicWidth"]),
        image_height = int(attrs["HighPicHeight"]),
        quality = attrs["Quality"],
        model_name = attrs["Model_Name"],
    )

    return product


@click.command(help="Import product data.")
@click.option(
    "-d",
    "--db-auth-path",
    type=click.STRING,
    required=True,
)
@click.option(
    "-f",
    "--file-path",
    type=click.STRING,
    required=True,
)
def main(
    db_auth_path: str,
    file_path: str
):
    repo = Repository(db_auth_path=db_auth_path)
    for event, element in etree.iterparse(file_path, events=("start", "end"), tag="file"):
        if event == "start":
            product_file_name = Path(element.get("path")).name
            product_file_path = Path(file_path).parent / product_file_name
            vals = parse_product_info(product_file_path)

            product_id = element.get("Product_ID")
            if repo.product_exists(product_id):
                continue

            product = extract_product_info(element)
            repo.create_product(product)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
