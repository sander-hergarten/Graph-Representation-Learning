import pytest
from dotenv import load_dotenv, find_dotenv


@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    load_dotenv(find_dotenv())
