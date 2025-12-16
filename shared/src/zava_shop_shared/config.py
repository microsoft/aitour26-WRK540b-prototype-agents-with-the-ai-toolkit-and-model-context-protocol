import logging
import os
import pathlib
import re

# Load environment variables from .env file if dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass

# Configure basic logging to show INFO level messages
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")

# Suppress verbose Azure Application Insights logging (same as sales_analysis.py)
for name in [
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.ai.agents",
    "azure.ai.projects",
    "azure.core",
    "azure.identity",
    "uvicorn.access",
    "azure.monitor.opentelemetry.exporter.export._base",
]:
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class Config:
    """Configuration class for managing application settings."""

    def __init__(self):
        """Initialize configuration with environment variables."""

        ABS_DB_PATH = "sqlite+aiosqlite:////workspace/data/retail.db"
        
        # Use absolute path if running in container (/workspace exists), else compute absolute path
        if pathlib.Path("/workspace").exists():
            DEFAULT_SQLITE_URL = ABS_DB_PATH
        else:
            # Compute absolute path to data/retail.db from the shared module location
            shared_dir = pathlib.Path(__file__).parent.parent.parent.parent
            db_path = (shared_dir / "data" / "retail.db").resolve()
            DEFAULT_SQLITE_URL = f"sqlite+aiosqlite:///{db_path}"

        # SQLite database URL
        self._sqlite_database_url: str = self._clean_env_value(os.getenv("SQLITE_DATABASE_URL", DEFAULT_SQLITE_URL))

        # Load and clean Application Insights connection string
        appinsights_raw = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
        self._appinsights_connection_string: str = self._clean_env_value(appinsights_raw)

        # Always log configuration info
        self._log_config_info()

    def _clean_env_value(self, value: str) -> str:
        """Strip surrounding quotes that might be added by Docker."""
        return value.strip('"').strip("'") if value else ""

    def _log_config_info(self) -> None:
        """Log configuration information."""

        logger.info(
            "APPLICATIONINSIGHTS_CONNECTION_STRING: '%s'",
            self._appinsights_connection_string,
        )

    @property
    def sqlite_database_url(self) -> str:
        """Returns the SQLite database URL."""
        return self._sqlite_database_url

    @property
    def applicationinsights_connection_string(self) -> str:
        """
        Returns the Application Insights connection string with cleaned endpoint URLs.
        Ensures endpoint URLs do not have trailing slashes.
        """
        if not self._appinsights_connection_string:
            return ""

        # Remove trailing slashes from IngestionEndpoint and LiveEndpoint
        connection_string = re.sub(
            r"(IngestionEndpoint|LiveEndpoint)=([^;]+)/",
            r"\1=\2",
            self._appinsights_connection_string,
        )
        return connection_string
