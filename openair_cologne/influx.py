"""Interactions with influx database."""

from influxdb import DataFrameClient


def get_db_client() -> DataFrameClient:
    client = DataFrameClient(host='openair.cologne.codefor.de/influx',
                             port=443,
                             database='lanuv')
    return client


def query_influx(query: str) -> dict:
    client = get_db_client()
    result = client.query(query)
    client.close()
    return result
