from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class RecommendationType(str, Enum):
    PRODUCT = "product"
    CATEGORY = "category"
    BRAND = "brand"

class RecommendationRequest(BaseModel):
    customer_id: int = Field(..., description="The ID of the customer to get recommendations for")
    num_recommendations: int = Field(5, ge=1, le=20, description="Number of recommendations to return")

class ProductRecommendation(BaseModel):
    id: int = Field(..., description="The ID of the recommended product")
    name: str = Field(..., description="The name of the recommended product")
    price: float = Field(..., description="The price of the recommended product")
    category: str = Field(..., description="The category of the recommended product")
    brand: str = Field(..., description="The brand of the recommended product")
    rating: float = Field(..., ge=0, le=5, description="The average rating of the recommended product")

class RecommendationResult(BaseModel):
    customer_id: int = Field(..., description="The ID of the customer the recommendations are for")
    recommendations: List[ProductRecommendation] = Field(..., description="List of recommended products")
    explanation: str = Field(..., description="Explanation for the recommendations")

class RecommendationResponse(RecommendationResult):
    pass

class BatchRecommendationRequest(BaseModel):
    customer_ids: List[int] = Field(..., description="List of customer IDs to get recommendations for")
    num_recommendations: int = Field(5, ge=1, le=20, description="Number of recommendations to return for each customer")

class RecommendationModel(BaseModel):
    id: str = Field(..., description="Unique identifier for the recommendation model")
    name: str = Field(..., description="Name of the recommendation model")
    description: str = Field(..., description="Description of the recommendation model")

class RecommendationFeedbackSchema(BaseModel):
    customer_id: int = Field(..., description="The ID of the customer providing feedback")
    product_id: int = Field(..., description="The ID of the product being rated")
    rating: float = Field(..., ge=1, le=5, description="The rating given by the customer")

class PerformanceMetrics(BaseModel):
    click_through_rate: float = Field(..., ge=0, le=1, description="Click-through rate of recommendations")
    average_rating: float = Field(..., ge=0, le=5, description="Average rating of recommended products")
    conversion_rate: float = Field(..., ge=0, le=1, description="Conversion rate of recommendations")

class TrendingProductResponse(BaseModel):
    id: int = Field(..., description="The ID of the trending product")
    name: str = Field(..., description="The name of the trending product")
    category: str = Field(..., description="The category of the trending product")
    brand: str = Field(..., description="The brand of the trending product")
    order_count: int = Field(..., description="Number of orders for this product")

class PersonalizedDeal(BaseModel):
    product: ProductRecommendation
    original_price: float = Field(..., description="Original price of the product")
    discounted_price: float = Field(..., description="Discounted price of the product")
    discount_percentage: float = Field(..., ge=0, le=100, description="Discount percentage")

class PersonalizedDealResponse(PersonalizedDeal):
    pass

class RecommendationLog(BaseModel):
    id: int = Field(..., description="Unique identifier for the recommendation log")
    customer_id: int = Field(..., description="The ID of the customer who received the recommendation")
    product_id: int = Field(..., description="The ID of the recommended product")
    timestamp: datetime = Field(..., description="Timestamp of when the recommendation was made")
    recommendation_type: str = Field(..., description="Type of recommendation (e.g., 'product', 'batch')")

class RecommendationFeedback(BaseModel):
    id: int = Field(..., description="Unique identifier for the feedback")
    customer_id: int = Field(..., description="The ID of the customer who provided the feedback")
    product_id: int = Field(..., description="The ID of the product that was rated")
    rating: float = Field(..., ge=1, le=5, description="The rating given by the customer")
    timestamp: datetime = Field(..., description="Timestamp of when the feedback was provided")

class ModelTrainingStatus(BaseModel):
    status: str = Field(..., description="Current status of model training")
    start_time: datetime = Field(..., description="Start time of the training process")
    end_time: Optional[datetime] = Field(None, description="End time of the training process")
    error_message: Optional[str] = Field(None, description="Error message if training failed")

class RecommendationExplanation(BaseModel):
    reason: str = Field(..., description="Reason for the recommendation")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score for this reason")

class DetailedRecommendationResponse(RecommendationResponse):
    explanations: List[RecommendationExplanation] = Field(..., description="Detailed explanations for each recommendation")

class RecommendationSettings(BaseModel):
    use_collaborative_filtering: bool = Field(..., description="Whether to use collaborative filtering")
    use_content_based: bool = Field(..., description="Whether to use content-based filtering")
    min_similarity_score: float = Field(..., ge=0, le=1, description="Minimum similarity score for recommendations")
    max_recommendations: int = Field(..., ge=1, description="Maximum number of recommendations to generate")

class CategoryAffinityScore(BaseModel):
    category: str = Field(..., description="Product category")
    affinity_score: float = Field(..., ge=0, le=1, description="Affinity score for the category")

class CustomerRecommendationProfile(BaseModel):
    customer_id: int = Field(..., description="The ID of the customer")
    favorite_categories: List[str] = Field(..., description="List of the customer's favorite product categories")
    category_affinity_scores: List[CategoryAffinityScore] = Field(..., description="Affinity scores for different product categories")
    last_recommendation_date: datetime = Field(..., description="Date of the last recommendation made to this customer")
    total_recommendations: int = Field(..., ge=0, description="Total number of recommendations made to this customer")
    average_recommendation_rating: float = Field(..., ge=0, le=5, description="Average rating of recommendations for this customer")
