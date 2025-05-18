import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--generate-missing",
        action="store_true",
        help="Generate expected output files if they don't exist"
    )

@pytest.fixture
def generate_missing(request):
    return request.config.getoption("--generate-missing") 