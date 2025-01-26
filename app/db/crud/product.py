from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from app.models.product import Product
from app.schemas.product import ProductCreate, ProductUpdate
from app.core.logging import logger
from typing import List, Optional
from datetime import datetime

def create_product(db: Session, product: ProductCreate) -> Product:
    try:
        db_product = Product(**product.dict())
        db.add(db_product)
        db.commit()
        db.refresh(db_product)
        logger.info(f"Created product with ID: {db_product.id}")
        return db_product
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error creating product: {str(e)}")
        raise

def get_product(db: Session, product_id: int) -> Optional[Product]:
    try:
        product = db.query(Product).filter(Product.id == product_id).first()
        if product:
            logger.info(f"Retrieved product with ID: {product_id}")
        else:
            logger.info(f"Product with ID {product_id} not found")
        return product
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving product: {str(e)}")
        raise

def get_products(db: Session, skip: int = 0, limit: int = 100) -> List[Product]:
    try:
        products = db.query(Product).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(products)} products")
        return products
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving products: {str(e)}")
        raise

def update_product(db: Session, product_id: int, product: ProductUpdate) -> Optional[Product]:
    try:
        db_product = db.query(Product).filter(Product.id == product_id).first()
        if db_product:
            update_data = product.dict(exclude_unset=True)
            for key, value in update_data.items():
                setattr(db_product, key, value)
            db_product.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(db_product)
            logger.info(f"Updated product with ID: {product_id}")
            return db_product
        else:
            logger.info(f"Product with ID {product_id} not found for update")
            return None
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error updating product: {str(e)}")
        raise

def delete_product(db: Session, product_id: int) -> bool:
    try:
        db_product = db.query(Product).filter(Product.id == product_id).first()
        if db_product:
            db.delete(db_product)
            db.commit()
            logger.info(f"Deleted product with ID: {product_id}")
            return True
        else:
            logger.info(f"Product with ID {product_id} not found for deletion")
            return False
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error deleting product: {str(e)}")
        raise

def get_products_by_category(db: Session, category: str, skip: int = 0, limit: int = 100) -> List[Product]:
    try:
        products = db.query(Product).filter(Product.category == category).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(products)} products in category: {category}")
        return products
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving products by category: {str(e)}")
        raise

def get_top_selling_products(db: Session, limit: int = 10) -> List[Product]:
    try:
        products = db.query(Product).order_by(Product.sales_count.desc()).limit(limit).all()
        logger.info(f"Retrieved top {len(products)} selling products")
        return products
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving top selling products: {str(e)}")
        raise

def update_product_inventory(db: Session, product_id: int, quantity_change: int) -> Optional[Product]:
    try:
        db_product = db.query(Product).filter(Product.id == product_id).first()
        if db_product:
            db_product.inventory_count += quantity_change
            db_product.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(db_product)
            logger.info(f"Updated inventory for product with ID: {product_id}")
            return db_product
        else:
            logger.info(f"Product with ID {product_id} not found for inventory update")
            return None
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error updating product inventory: {str(e)}")
        raise
