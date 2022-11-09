import pytest


def pytest_addoption(parser):
    """
    Add execution options to the commandline
    :param parser:
    :return:
    """

    parser.addoption(
        "--runintegration", action="store_true", default=False, help="run integration tests"
    )


def pytest_configure(config):
    """
    Add pytext init lines
    :param config:
    :return:
    """
    config.addinivalue_line("markers", "integration: Mark test as integration.")


def pytest_collection_modifyitems(config, items):
    """
    Modify the tests to skip integration unless specified
    :param config:
    :param items:
    :return:
    """

    # Determine if any markers need to be skipped.
    if config.getoption("--runintegration"):
        # --runintegration given in cli: do not skip integration tests
        return

    markers_skip_integration = pytest.mark.skip(reason="need --runintegration option to run")

    for item in items:
        if "integration" in item.keywords:
            item.add_marker(markers_skip_integration)
