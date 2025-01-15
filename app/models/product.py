from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from app.db.base_class import Base

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    sku = Column(String, unique=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    price = Column(Float)
    cost = Column(Float)
    category = Column(String, index=True)
    subcategory = Column(String)
    brand = Column(String, index=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"))
    stock_quantity = Column(Integer)
    reorder_point = Column(Integer)
    weight = Column(Float)
    dimensions = Column(JSONB)  # e.g., {"length": 10, "width": 5, "height": 2}
    color = Column(String)
    size = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    tags = Column(ARRAY(String))
    attributes = Column(JSONB)
    image_urls = Column(ARRAY(String))

    # Relationships
    supplier = relationship("Supplier", back_populates="products")
    order_items = relationship("OrderItem", back_populates="product")

    def __repr__(self):
        return f"<Product {self.name}>"

    @property
    def profit_margin(self):
        return (self.price - self.cost) / self.price * 100 if self.price > 0 else 0

    def update_stock(self, quantity_change):
        self.stock_quantity += quantity_change
        if self.stock_quantity < self.reorder_point:
            # Trigger reorder process
            self.reorder()

    def reorder(self):
        # Logic for reordering product
        pass

    def update_price(self, new_price):
        self.price = new_price
        self.updated_at = datetime.utcnow()

    def add_tag(self, tag):
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag):
        if tag in self.tags:
            self.tags.remove(tag)

    def update_attributes(self, new_attributes):
        self.attributes.update(new_attributes)
        self.updated_at = datetime.utcnow()
