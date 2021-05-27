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
        for node in product_element.findall("EANCode")
    ]


def parse_features(product_element) -> List[ProductFeature]:
    return [
        ProductFeature(
            id=node.Feature.get("ID"),
            category_id=node.get("CategoryFeature_ID"),
            group_id=node.get("CategoryFeatureGroup_ID"),
            measure_id=node.LocalValue.Measure.get("ID"),
            name=node.Feature.Name.get("Value"),
            value=node.LocalValue.get("Value"),
            presentation_value=node.get("Presentation_Value"),
            is_translated=node.get("Translated") == "1",
            is_mandatory=node.get("Mandatory") == "1",
            is_searchable=node.get("Searchable") == "1",
        )
        for node in product_element.findall("ProductFeature")
    ]


def parse_images(product_element) -> List[ProductImage]:
    return [
        ProductImage(
            type=node.get("Type"),
            is_main=node.get("IsMain") == "Y",
            url=node.get("Original"),
            size_bytes=int(node.get("OriginalSize")),
            updated_at=to_datetime(node, ".", "UpdatedDate", "%Y-%m-%d %H:%M:%S"),
        )
        for node in product_element.findall("ProductGallery/ProductPicture")
    ]


def parse_variant_ids(variant_element) -> List[ProductVariantId]:
    return [
        ProductVariantId(type=node.get("Type"), value=node.get("Value"))
        for node in variant_element.findall("VariantIdentifiers/Identifier")
    ]


def parse_variants(product_element) -> List[ProductVariant]:
    return [
        ProductVariant(
            id=node.get("ID"),
            description=node.get("Desc"),
            identifiers=parse_variant_ids(node)
        )
        for node in product_element.findall("Variants/Variant")
    ]


def to_datetime(xml_element, xpath: str, attr_name: str, format: str) -> Optional[datetime]:
    node = xml_element.find(xpath)
    if node is None:
        return None
    val = node.get(attr_name)
    if val is not None and len(val) > 0:
        return datetime.strptime(val, format)
    return None


def parse_product_info(file_path: Path, product: Product) -> List[MultiValue]:
    print(f"Extracting product info from {file_path}")
    parse_xml = objectify.parse(str(file_path))
    product_node = parse_xml.getroot().Product

    if product_node is None:
        print(f' > Skipped due to wrong formatting.')
        return

    err_msg = product_node.get("ErrorMessage")
    if err_msg is not None:
        print(f' > Skipped due error message: "{err_msg}".')
        return

    product.released_on = to_datetime(product_node, ".", "ReleaseDate", "%Y-%m-%d")
    product.end_of_life_on = to_datetime(product_node, "EndOfLifeDate/Date", "Value", "%Y-%m-%d")
    
    product.title = product_node.get("Title")
    product.category_name = product_node.Category.Name.get("Value")
    product.supplier_name = product_node.Supplier.get("Name")

    product.description_short = product_node.ProductDescription.get("ShortDesc")
    product.description_middle = product_node.ProductDescription.get("MiddleDesc")
    product.description_long = product_node.ProductDescription.get("LongDesc")

    product.summary_short = product_node.SummaryDescription.ShortSummaryDescription.text
    product.summary_long = product_node.SummaryDescription.LongSummaryDescription.text

    product.warranty = product_node.ProductDescription.get("WarrantyInfo")
    product.url_details = product_node.ProductDescription.get("URL")
    product.url_manual = product_node.ProductDescription.get("ManualPDFURL")
    product.url_pdf = product_node.ProductDescription.get("PDFURL")

    product.ean = parse_ean(product_node)
    product.features = parse_features(product_node)
    product.images = parse_images(product_node)
    product.variants = parse_variants(product_node)


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
    skipped: bool = False
    for event, element in etree.iterparse(file_path, events=("start", "end"), tag=("file", "Country_Market")):
        if event == "start" and element.tag == "file":
            product_id = element.get("Product_ID")
            skipped = repo.product_exists(product_id)
            
            if not skipped:
                product = extract_product_info(element)
        elif event == "start" and element.tag == "Country_Market":
            if not skipped:
                market = element.get("Value")
                product.country_markets.append(market)
        elif event == "end" and element.tag == "file":
            if not skipped:
                file_name = f"{product.id}.xml"
                product_file_path = Path(file_path).parent / file_name
                parse_product_info(product_file_path, product)
                repo.create_product(product)
                print(f"Product {product.id} created successfully!")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
