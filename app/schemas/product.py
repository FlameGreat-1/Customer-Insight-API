from pydantic import BaseModel, validator, constr
from typing import Optional, List, Dict
from datetime import datetime

class ProductBase(BaseModel):
    sku: constr(min_length=1, max_length=50)
    name: constr(min_length=1, max_length=100)
    description: Optional[str]
    price: float
    cost: float
    category: str
    subcategory: Optional[str]
    brand: str
    supplier_id: int
    stock_quantity: int
    reorder_point: int
    weight: Optional[float]
    dimensions: Optional[Dict[str, float]]
    color: Optional[str]
    size: Optional[str]
    is_active: bool = True
    tags: Optional[List[str]]
    attributes: Optional[Dict]
    image_urls: Optional[List[str]]

    @validator('price', 'cost')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Price and cost must be positive')
        return v

class ProductCreate(ProductBase):
    pass

class ProductUpdate(ProductBase):
    sku: Optional[constr(min_length=1, max_length=50)]
    name: Optional[constr(min_length=1, max_length=100)]
    price: Optional[float]
    cost: Optional[float]
    category: Optional[str]
    brand: Optional[str]
    supplier_id: Optional[int]
    stock_quantity: Optional[int]
    reorder_point: Optional[int]

class ProductInDB(ProductBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class ProductOut(ProductInDB):
    profit_margin: float

    @validator('profit_margin', pre=True, always=True)
    def calculate_profit_margin(cls, v, values):
        if 'price' in values and 'cost' in values:
            return (values['price'] - values['cost']) / values['price'] * 100 if values['price'] > 0 else 0
        return v

class ProductWithRelations(ProductOut):
    supplier: 'SupplierOut'
    order_items: List['OrderItemOut']

    class Config:
        orm_mode = True

# Circular import resolution
from .supplier import SupplierOut
from .order import OrderItemOut

ProductWithRelations.update_forward_refs()
