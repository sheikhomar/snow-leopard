from typing import List
from datetime import datetime
from pathlib import Path

import click
from lxml import etree, objectify
from app.models import *


def import_products(db_auth_path: str, import_path: str) -> None:
    
    import_dir_path = Path(import_path)
    if not import_dir_path.exists():
        print(f"Path {import_path} does not exist!")
        return

    repo = Repository(db_auth_path=db_auth_path)
    if import_dir_path.is_dir():
        file_paths = import_dir_path.glob("*.xml")
        for fp in file_paths:
            parse_product_info(fp)
    else:
        parse_product_info(import_dir_path)


def parse_ean(product_element) -> List[ProductEAN]:
    return [
        ProductEAN(variant_id=node.get("VariantID"), ean=node.get("EAN"))
        for node in product_element.EANCode
    ]


def extract_image_info(xml_element, type: str) -> ProductImage:
    return ProductImage(
        type=type.lower(),
        url=xml_element.get(f"{type}Pic"),
        width=int(xml_element.get(f"{type}PicWidth")),
        height=int(xml_element.get(f"{type}PicHeight")),
        size_bytes=int(xml_element.get(f"{type}PicSize"))
    )


def parse_product_info(file_path: Path, product: Product) -> List[MultiValue]:
    print(f"Extracting product info from {file_path}")
    parse_xml = objectify.parse(str(file_path))
    product_node = parse_xml.getroot().Product
    product.images.append(extract_image_info(product_node, "High"))
    product.images.append(extract_image_info(product_node, "Low"))
    product.released_on = datetime.strptime(product_node.get("ReleaseDate"), "%Y-%m-%d")
    product.end_of_life_on = datetime.strptime(product_node.EndOfLifeDate.Date.get("Value"), "%Y-%m-%d")
    product.title = product_node.get("Title")
    product.category_name = product_node.Category.Name.get("Value")

    product.description_short = product_node.ProductDescription.get("ShortDesc")
    product.description_middle = product_node.ProductDescription.get("MiddleDesc")
    product.description_long = product_node.ProductDescription.get("LongDesc")

    product.warranty = product_node.ProductDescription.get("WarrantyInfo")
    product.url_details = product_node.ProductDescription.get("URL")
    product.url_manual = product_node.ProductDescription.get("ManualPDFURL")
    product.url_pdf = product_node.ProductDescription.get("PDFURL")

    product.ean = parse_ean(product_node)


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
    product: Product = None
    for event, element in etree.iterparse(file_path, events=("start", "end"), tag=("file", "Country_Market")):
        if event == "start" and element.tag == "file":
            product_id = element.get("Product_ID")
            if repo.product_exists(product_id):
                continue
            product = extract_product_info(element)
        elif event == "start" and element.tag == "Country_Market":
            market = element.get("Value")
            product.country_markets.append(market)
        elif event == "end" and element.tag == "file":
            file_name = f"{product.id}.xml"
            product_file_path = Path(file_path).parent / file_name
            parse_product_info(product_file_path, product)
            repo.create_product(product)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
