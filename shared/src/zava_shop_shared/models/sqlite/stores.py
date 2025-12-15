"""
Store model for SQLite
"""

from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.orm import relationship

from .base import Base


class Store(Base):
    """Represents a store location"""

    __tablename__ = "stores"

    store_id = Column(Integer, primary_key=True, autoincrement=True)
    store_name = Column(String, nullable=False, unique=True)
    rls_user_id = Column(String, nullable=False)
    is_online = Column(Boolean, default=False, nullable=False)

    # Relationships
    orders = relationship("Order", back_populates="store")
    inventory = relationship("Inventory", back_populates="store")
    order_items = relationship("OrderItem", back_populates="store")
    customers = relationship(
        "Customer", foreign_keys="Customer.primary_store_id", overlaps="primary_store"
    )

    def __repr__(self):
        return f"<Store(id={self.store_id}, name='{self.store_name}', online={self.is_online})>"
