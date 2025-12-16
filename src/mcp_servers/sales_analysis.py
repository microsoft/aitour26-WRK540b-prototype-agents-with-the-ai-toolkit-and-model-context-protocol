#!/usr/bin/env python3
"""
Provides comprehensive customer sales database access with individual table schema tools for Zava Retail DIY Business.
SQLite Edition - uses pure Python for semantic search.
"""

import asyncio
import json
import logging
import math
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, AsyncIterator, Optional

from azure.monitor.opentelemetry import configure_azure_monitor
from fastmcp import FastMCP
from pydantic import Field
from sqlalchemy import and_, func, select, text
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from zava_shop_shared.config import Config
from zava_shop_shared.finance_sqlite import FinanceSQLiteProvider
from zava_shop_shared.models.sqlite import (
    Category,
    Customer,
    Inventory,
    Order,
    OrderItem,
    Product,
    ProductDescriptionEmbedding,
    ProductType,
    Store,
    Supplier,
)

# Import Azure OpenAI for embeddings
# Support both running as module (-m mcp_servers.sales_analysis) and directly (python sales_analysis.py)
try:
    from mcp_servers.sales_analysis_text_embeddings import SemanticSearchTextEmbedding
    USE_REAL_EMBEDDINGS = True
except ImportError:
    try:
        from sales_analysis_text_embeddings import SemanticSearchTextEmbedding
        USE_REAL_EMBEDDINGS = True
    except ImportError:
        USE_REAL_EMBEDDINGS = False

config = Config()

logger = logging.getLogger(__name__)

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


db_provider = FinanceSQLiteProvider()

# Initialize semantic search provider (real or fake)
semantic_search_provider = SemanticSearchTextEmbedding() if USE_REAL_EMBEDDINGS else None


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator:
    # Initialize database connection once at startup
    await db_provider.open()
    logger.info("Database connection initialized")
    yield
    await db_provider.close_engine()


# Create MCP server
mcp = FastMCP("mcp-zava-sales", lifespan=app_lifespan, stateless_http=True)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(_request: Request) -> Response:
    """Health check endpoint."""
    return JSONResponse({"status": "ok"})


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


async def _semantic_search_products_impl(
    query_description: str,
    limit: int = 20,
    min_similarity: float = 0.3,
) -> list[dict]:
    """Implementation of semantic search for products."""
    try:
        logger.info(f"Searching products with query: '{query_description}', limit: {limit}")

        # Generate embedding for the query using Azure OpenAI
        if not (USE_REAL_EMBEDDINGS and semantic_search_provider and semantic_search_provider.is_available()):
            logger.error("Azure OpenAI not configured. Semantic search requires Azure OpenAI embeddings.")
            return []

        logger.info("Using Azure OpenAI for embeddings")
        query_embedding = semantic_search_provider.generate_query_embedding(query_description)
        if not query_embedding:
            logger.error("Failed to generate embedding from Azure OpenAI")
            return []

        async with db_provider.get_session() as session:
            # Fetch all product embeddings with supplier info
            stmt = (
                select(
                    Product.product_id,
                    Product.sku,
                    Product.product_name,
                    Product.product_description,
                    Product.base_price,
                    Product.cost,
                    Product.discontinued,
                    Category.category_name,
                    ProductType.type_name,
                    ProductDescriptionEmbedding.description_embedding,
                    Supplier.supplier_id,
                    Supplier.supplier_name,
                    Supplier.supplier_rating,
                    Supplier.lead_time_days,
                    Supplier.minimum_order_amount,
                    Supplier.bulk_discount_percent,
                )
                .select_from(Product)
                .join(ProductDescriptionEmbedding, Product.product_id == ProductDescriptionEmbedding.product_id)
                .join(Category, Product.category_id == Category.category_id)
                .join(ProductType, Product.type_id == ProductType.type_id)
                .join(Supplier, Product.supplier_id == Supplier.supplier_id)
                .where(Product.discontinued == False)
            )

            result = await session.execute(stmt)
            rows = result.all()

            # Calculate similarities using pure Python
            scored_products = []
            for row in rows:
                # Parse embedding from JSON string
                product_embedding = json.loads(row.description_embedding)
                similarity = cosine_similarity(query_embedding, product_embedding)

                if similarity >= min_similarity:
                    scored_products.append(
                        {
                            "product_id": row.product_id,
                            "sku": row.sku,
                            "product_name": row.product_name,
                            "product_description": row.product_description,
                            "base_price": float(row.base_price),
                            "cost": float(row.cost),
                            "category_name": row.category_name,
                            "type_name": row.type_name,
                            "similarity_score": round(similarity, 4),
                            "supplier": {
                                "supplier_id": row.supplier_id,
                                "supplier_name": row.supplier_name,
                                "supplier_rating": float(row.supplier_rating),
                                "lead_time_days": row.lead_time_days,
                                "minimum_order_amount": float(row.minimum_order_amount),
                                "bulk_discount_percent": float(row.bulk_discount_percent),
                            },
                        }
                    )

            # Sort by similarity (descending) and limit
            scored_products.sort(key=lambda x: x["similarity_score"], reverse=True)
            return scored_products[:limit]

    except Exception as e:
        logger.error(f"Error in semantic_search_products: {e}")
        return []


@mcp.tool()
async def semantic_search_products(
    query_description: Annotated[
        str,
        Field(
            description="Zava product you're looking for using natural language. Include purpose, features, or use case. For example: 'waterproof electrical box for outdoor use', '15 amp circuit breaker', or 'LED light bulbs for kitchen ceiling'."
        ),
    ],
    limit: Annotated[int, Field(description="Maximum number of results to return")] = 20,
    min_similarity: Annotated[float, Field(description="Minimum similarity score (0.0 to 1.0)")] = 0.3,
) -> list[dict]:
    """
    **ALWAYS USE THIS TOOL** for queries related to finding, listing, or describing Zava products based on **features, usage, or natural language descriptions**.

    Uses vector similarity search against product description embeddings to find
    products that match the natural language query. Returns products ranked by
    semantic relevance with their descriptions, categories, and basic info.

    Args:
        query_description: Natural language description of what you're looking for
        limit: Maximum number of products to return (default: 20)
        min_similarity: Minimum similarity threshold 0.0-1.0 (default: 0.3)

    Returns:
        List of products with similarity scores, product id, product sku, descriptions, categories and supplier info.

    Example:
        >>> results = await semantic_search_products(
        >>>     query_description="cordless drill with battery",
        >>>     limit=5
        >>> )
    """
    return await _semantic_search_products_impl(query_description, limit, min_similarity)


@mcp.tool()
async def get_database_schema() -> str:
    """
    Retrieve schemas for all supported tables.

    Returns:
        Concatenated schema strings for all tables with column information.
    """

    logger.info("Retrieving schemas for all tables")

    try:
        table_map = {
            "customers": Customer,
            "stores": Store,
            "categories": Category,
            "product_types": ProductType,
            "products": Product,
            "orders": Order,
            "order_items": OrderItem,
            "inventory": Inventory,
        }

        schemas = []
        async with db_provider.get_session() as session:
            for table_name, model in table_map.items():
                columns_info = []
                for column in model.__table__.columns:
                    col_type = str(column.type)
                    nullable = "NULL" if column.nullable else "NOT NULL"
                    primary = "PRIMARY KEY" if column.primary_key else ""
                    columns_info.append(f"  - {column.name}: {col_type} {nullable} {primary}".strip())

                schema_str = f"# Table: {table_name}\n\n**Columns:**\n" + "\n".join(columns_info) + "\n"
                schemas.append(schema_str)

        return "\n".join(schemas)

    except Exception as e:
        logger.error("Error retrieving table schemas: %s", e)
        return f"Error retrieving table schemas: {e!s}"


@mcp.tool()
async def execute_sales_query(sql_query: Annotated[str, Field(description="A well-formed SQLite query.")]) -> str:
    """
    Use this tool **ONLY** for fetching quantitative, sales, inventory, pricing, or store-related data.

    **CRITICAL PRE-REQUISITE (MUST READ):**
    Before generating the `sql_query` argument, you **MUST** call `get_database_schema` to verify exact table and column names.
    * **NEVER** guess or hallucinate table/column names.
    * **NEVER** skip the schema check, even if you think you know the schema.
    * If you generate SQL with invalid column names, the query will fail.

    **CONSTRAINT: Product Discovery**
    **NEVER** use this tool to find products based on features, descriptions, or names; use `semantic_search_products` for that.

    **OUTPUT RULES:**
    * Join related tables (e.g., stores, products) to return readable names, NOT IDs.
    * **NEVER** include raw entity IDs (e.g., `product_id`, `order_id`) in the final JSON; they are useless to the user.
    * To identify store types, use `stores.is_online`: 1 = Online, 0 = Physical.
    * **ALWAYS** add `LIMIT 20` to your query.

    Args:
        sql_query: A well-formed SQLite query valid against the verified schema.

    Returns:
        Query results as a JSON string.
    """
    logger.info("Executing SQLite query: %s", sql_query)

    try:
        if not sql_query:
            return "Error: sql_query parameter is required"

        async with db_provider.get_session() as session:
            result = await session.execute(text(sql_query))
            rows = result.fetchall()

            if not rows:
                return "Query executed successfully. No results returned."

            # Convert to list of dicts
            columns = result.keys()
            results = []
            for row in rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    value = row[i]
                    # Convert Decimal to float for JSON serialization
                    if hasattr(value, "__float__"):
                        value = float(value)
                    row_dict[col] = value
                results.append(row_dict)

            return json.dumps(results, indent=2, default=str)

    except Exception as e:
        logger.error("Error executing database query: %s", e)
        return f"Error executing database query: {e!s}"


@mcp.tool()
async def get_current_utc_date() -> str:
    """
    Get the current UTC date and time in ISO format.

    **USAGE:** Call this tool FIRST when the user mentions relative timeframes like "today", "yesterday", "last month", or "current quarter" to calculate exact date ranges for SQL queries.

    Returns:
        Current UTC date and time in ISO format (YYYY-MM-DDTHH:MM:SS.fffffZ)
    """
    logger.info("Retrieving current UTC date and time")
    try:
        current_utc = datetime.now(UTC)
        return f"Current UTC Date/Time: {current_utc.isoformat()}"
    except Exception as e:
        logger.error("Error retrieving current UTC date: %s", e)
        return f"Error retrieving current UTC date: {e!s}"


async def test_semantic_search():
    """Test function to demonstrate semantic search."""
    print("\n" + "=" * 70)
    print("SEMANTIC SEARCH TEST - Using Azure OpenAI Embeddings")
    print("=" * 70)

    queries = ["test query 1", "test query 2", "test query 3"]

    for query in queries:
        print(f'\n🔍 Query: "{query}"')
        print("-" * 70)

        results = await _semantic_search_products_impl(query_description=query, limit=5, min_similarity=0.0)

        if results:
            print(f"✅ Found {len(results)} products:\n")
            for i, product in enumerate(results, 1):
                print(f"{i}. {product['product_name']}")
                print(f"   SKU: {product['sku']}")
                print(f"   💰 ${product['base_price']:.2f}")
                print(f"   📂 {product['category_name']} - {product['type_name']}")
                print(f"   📊 Similarity: {product['similarity_score']:.4f}")
                print()
        else:
            print("❌ No products found")

    print("=" * 70)
    print("\n✨ Note: All queries use the SAME random embedding from the database,")
    print("   so they match the same products with high similarity scores.")


if __name__ == "__main__":
    # Check if running in test mode
    if os.getenv("TEST_SEMANTIC_SEARCH"):
        asyncio.run(test_semantic_search())
    else:
        logger.info("🚀 Starting Sales Analysis MCP Server (SQLite Edition)")

        # Configure Azure Monitor if connection string is available
        if config.applicationinsights_connection_string:
            configure_azure_monitor(connection_string=config.applicationinsights_connection_string)

        # Configure server settings
        port = int(os.getenv("PORT", 8004))
        host = os.getenv("HOST", "0.0.0.0")

        logger.info(
            "❤️ 📡 Sales Analysis MCP endpoint starting at: http://%s:%d/mcp",
            host,
            port,
        )

        mcp.run(transport="http", host=host, port=port, path="/mcp", stateless_http=True)
