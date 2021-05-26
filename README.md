# Snow Leopard

A Python script to parse [IceCat Product Catalogue](https://icecat.co.uk/en).


## Getting Started

This projects relies on [pyenv](https://github.com/pyenv/pyenv) and [Poetry](https://python-poetry.org/docs/).

1. Install the required Python version:

   ```bash
   pyenv install
   ```

2. Install dependencies

   ```bash
   poetry install
   ```

3. Run ArangoDB via Docker:

    ```bash
    export IP=$(hostname -I | awk '{print $1}')
    docker volume create arangodb1
    docker run -it --name=adb1 --rm \
        -p 8528:8528 \
        -v arangodb1:/data \
        -v /var/run/docker.sock:/var/run/docker.sock \
        arangodb/arangodb-starter \
        --starter.address=$IP \
        --starter.mode=single
    ```

4. Insert data into the database:

    ```bash
    poetry run python -m app.data.import_products -d config/secrets/arangodb.json -f data/ice-cat-office-products/index.xml
    ```
