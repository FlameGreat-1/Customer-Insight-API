from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Index, CheckConstraint, func
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from app.db.base_class import Base
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import re
from app.core.config import settings
import boto3
from botocore.exceptions import ClientError
from app.models.order import Order
from app.models.order_item import OrderItem
from app.models.product_review import ProductReview
from app.utils.image_processing import compress_image, generate_thumbnail
from app.utils.currency_converter import convert_currency
from app.services.inventory_service import InventoryService
from app.services.notification_service import NotificationService
from app.services.analytics_service import AnalyticsService

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
    dimensions = Column(JSONB)
    color = Column(String)
    size = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    tags = Column(ARRAY(String))
    attributes = Column(JSONB)
    image_urls = Column(ARRAY(String))
    average_rating = Column(Float, default=0.0)
    total_reviews = Column(Integer, default=0)
    total_sales = Column(Integer, default=0)
    last_sold_at = Column(DateTime)
    view_count = Column(Integer, default=0)
    featured = Column(Boolean, default=False)
    discount_percentage = Column(Float, default=0.0)
    tax_rate = Column(Float, default=0.0)
    warranty_info = Column(String)
    return_policy = Column(String)
    seo_title = Column(String)
    seo_description = Column(String)
    seo_keywords = Column(ARRAY(String))

    # Relationships
    supplier = relationship("Supplier", back_populates="products")
    order_items = relationship("OrderItem", back_populates="product")
    reviews = relationship("ProductReview", back_populates="product")
    inventory_transactions = relationship("InventoryTransaction", back_populates="product")
    price_history = relationship("PriceHistory", back_populates="product")

    # Constraints
    __table_args__ = (
        CheckConstraint('price >= 0', name='check_price_positive'),
        CheckConstraint('cost >= 0', name='check_cost_positive'),
        CheckConstraint('stock_quantity >= 0', name='check_stock_quantity_positive'),
        CheckConstraint('discount_percentage >= 0 AND discount_percentage <= 100', name='check_discount_percentage_range'),
        Index('idx_product_name_brand', 'name', 'brand'),
        Index('idx_product_category_price', 'category', 'price'),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inventory_service = InventoryService()
        self.notification_service = NotificationService()
        self.analytics_service = AnalyticsService()

    def __repr__(self):
        return f"<Product {self.name} (SKU: {self.sku})>"

    @property
    def profit_margin(self) -> float:
        return (self.price - self.cost) / self.price * 100 if self.price > 0 else 0

    @property
    def discounted_price(self) -> float:
        return self.price * (1 - self.discount_percentage / 100)

    @validates('sku')
    def validate_sku(self, key, sku):
        if not re.match(r'^[A-Z0-9]{8,}$', sku):
            raise ValueError("SKU must be at least 8 characters long and contain only uppercase letters and numbers")
        return sku

    @validates('price', 'cost')
    def validate_price_cost(self, key, value):
        if value < 0:
            raise ValueError(f"{key.capitalize()} cannot be negative")
        return value

    def update_stock(self, quantity_change: int, transaction_type: str) -> None:
        self.stock_quantity += quantity_change
        if quantity_change < 0:
            self.last_sold_at = datetime.utcnow()
        self.inventory_service.record_transaction(self.id, quantity_change, transaction_type)
        if self.stock_quantity < self.reorder_point:
            self.reorder()

    def reorder(self) -> None:
        reorder_quantity = max(self.reorder_point * 2 - self.stock_quantity, 0)
        purchase_order = self.inventory_service.create_purchase_order(self.id, reorder_quantity)
        self.notification_service.notify_inventory_manager(purchase_order)

    def update_price(self, new_price: float) -> None:
        old_price = self.price
        self.price = new_price
        self.updated_at = datetime.utcnow()
        self.inventory_service.record_price_change(self.id, old_price, new_price)
        if new_price < old_price:
            self.notification_service.notify_price_drop(self)

    def add_tag(self, tag: str) -> None:
        if not self.tags:
            self.tags = []
        if tag not in self.tags:
            self.tags.append(tag)
            self.analytics_service.record_tag_addition(self.id, tag)

    def remove_tag(self, tag: str) -> None:
        if self.tags and tag in self.tags:
            self.tags.remove(tag)
            self.analytics_service.record_tag_removal(self.id, tag)

    def update_attributes(self, new_attributes: Dict[str, Any]) -> None:
        if not self.attributes:
            self.attributes = {}
        old_attributes = self.attributes.copy()
        self.attributes.update(new_attributes)
        self.updated_at = datetime.utcnow()
        self.analytics_service.record_attribute_change(self.id, old_attributes, self.attributes)

    def add_image(self, image_file: Any) -> None:
        compressed_image = compress_image(image_file)
        thumbnail = generate_thumbnail(compressed_image)
        image_url = self.upload_image_to_s3(compressed_image)
        thumbnail_url = self.upload_image_to_s3(thumbnail)
        if not self.image_urls:
            self.image_urls = []
        self.image_urls.append({"full": image_url, "thumbnail": thumbnail_url})

    def remove_image(self, image_url: str) -> None:
        if self.image_urls:
            self.image_urls = [img for img in self.image_urls if img['full'] != image_url]
        self.delete_image_from_s3(image_url)

    def update_rating(self, new_rating: float) -> None:
        self.average_rating = ((self.average_rating * self.total_reviews) + new_rating) / (self.total_reviews + 1)
        self.total_reviews += 1
        self.analytics_service.record_rating_update(self.id, self.average_rating, self.total_reviews)

    def record_sale(self, quantity: int = 1) -> None:
        self.total_sales += quantity
        self.last_sold_at = datetime.utcnow()
        self.analytics_service.record_sale(self.id, quantity)

    def record_view(self) -> None:
        self.view_count += 1
        self.analytics_service.record_product_view(self.id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sku": self.sku,
            "name": self.name,
            "description": self.description,
            "price": self.price,
            "discounted_price": self.discounted_price,
            "category": self.category,
            "subcategory": self.subcategory,
            "brand": self.brand,
            "stock_quantity": self.stock_quantity,
            "is_active": self.is_active,
            "tags": self.tags,
            "attributes": self.attributes,
            "image_urls": self.image_urls,
            "average_rating": self.average_rating,
            "total_reviews": self.total_reviews,
            "profit_margin": self.profit_margin,
            "featured": self.featured,
            "discount_percentage": self.discount_percentage,
            "warranty_info": self.warranty_info,
            "return_policy": self.return_policy,
            "seo_title": self.seo_title,
            "seo_description": self.seo_description,
            "seo_keywords": self.seo_keywords
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Product':
        return cls(**data)

    def is_low_stock(self) -> bool:
        return self.stock_quantity <= self.reorder_point

    def get_sales_velocity(self, days: int = 30) -> float:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        sales = OrderItem.query.join(Order).filter(
            OrderItem.product_id == self.id,
            Order.order_date.between(start_date, end_date)
        ).with_entities(func.sum(OrderItem.quantity)).scalar() or 0
        return sales / days

    def upload_image_to_s3(self, image_file: Any) -> Optional[str]:
        try:
            s3 = boto3.client('s3')
            bucket_name = settings.AWS_S3_BUCKET_NAME
            file_name = f"products/{self.sku}/{image_file.filename}"
            s3.upload_fileobj(image_file.file, bucket_name, file_name, ExtraArgs={'ACL': 'public-read'})
            return f"https://{bucket_name}.s3.amazonaws.com/{file_name}"
        except ClientError as e:
            self.notification_service.notify_admin("S3 Upload Error", str(e))
            return None

    def delete_image_from_s3(self, image_url: str) -> None:
        try:
            s3 = boto3.client('s3')
            bucket_name = settings.AWS_S3_BUCKET_NAME
            file_name = image_url.split(f"https://{bucket_name}.s3.amazonaws.com/")[1]
            s3.delete_object(Bucket=bucket_name, Key=file_name)
        except ClientError as e:
            self.notification_service.notify_admin("S3 Delete Error", str(e))

    def get_related_products(self, limit: int = 5) -> List['Product']:
        return Product.query.filter(
            Product.category == self.category,
            Product.id != self.id
        ).order_by(func.random()).limit(limit).all()

    def apply_discount(self, percentage: float) -> None:
        if 0 <= percentage <= 100:
            self.discount_percentage = percentage
            self.notification_service.notify_price_drop(self)
        else:
            raise ValueError("Discount percentage must be between 0 and 100")

    def get_price_history(self, days: int = 30) -> List[Dict[str, Any]]:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        return [
            {"date": ph.date, "price": ph.price}
            for ph in self.price_history
            if start_date <= ph.date <= end_date
        ]

    def get_reviews(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        reviews = ProductReview.query.filter(ProductReview.product_id == self.id).order_by(
            ProductReview.created_at.desc()
        ).offset(offset).limit(limit).all()
        return [review.to_dict() for review in reviews]

    def update_seo_info(self, title: str, description: str, keywords: List[str]) -> None:
        self.seo_title = title
        self.seo_description = description
        self.seo_keywords = keywords

    def get_inventory_transactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        transactions = self.inventory_transactions.order_by(
            InventoryTransaction.timestamp.desc()
        ).limit(limit).all()
        return [transaction.to_dict() for transaction in transactions]

    def get_price_in_currency(self, currency: str) -> float:
        return convert_currency(self.price, 'USD', currency)

    def is_bestseller(self) -> bool:
        return self.total_sales > 1000  # Arbitrary threshold, adjust as needed

    def calculate_reorder_point(self) -> None:
        # Dynamically calculate reorder point based on sales velocity and lead time
        sales_velocity = self.get_sales_velocity()
        lead_time = 7  # Assume 7 days lead time, adjust as needed
        safety_stock = sales_velocity * 3  # 3 days of safety stock
        self.reorder_point = int(sales_velocity * lead_time + safety_stock)

    def update_search_keywords(self) -> None:
        # Update search keywords based on product attributes
        keywords = set(self.name.split() + self.brand.split() + self.category.split() + self.subcategory.split())
        if self.attributes:
            for value in self.attributes.values():
                if isinstance(value, str):
                    keywords.update(value.split())
        self.seo_keywords = list(keywords)

    def get_total_revenue(self) -> float:
        return sum(item.quantity * item.price for item in self.order_items)

    def get_profit(self) -> float:
        return self.get_total_revenue() - (self.cost * self.total_sales)

    def is_trending(self) -> bool:
        recent_sales = OrderItem.query.join(Order).filter(
            OrderItem.product_id == self.id,
            Order.order_date >= datetime.utcnow() - timedelta(days=7)
        ).with_entities(func.sum(OrderItem.quantity)).scalar() or 0
        return recent_sales > 100  

