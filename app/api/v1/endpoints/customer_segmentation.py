from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.security import get_current_user
from app.schemas.customer_segmentation import (
    SegmentationRequest,
    SegmentationResponse,
    BatchSegmentationRequest,
    SegmentationModel,
    SegmentUpdateRequest
)
from app.services.customer_segmentation import CustomerSegmentationService
from app.models.user import User
from app.core.logging import logger
from typing import List
import asyncio

router = APIRouter()

@router.post("/segment", response_model=SegmentationResponse)
async def segment_customer(
    request: SegmentationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Customer segmentation requested by user {current_user.id}")
    service = CustomerSegmentationService(db)
    try:
        result = await service.segment_customer(request.customer_id)
        logger.info(f"Customer segmentation completed for user {current_user.id}")
        return SegmentationResponse(
            customer_id=request.customer_id,
            segment=result.segment,
            confidence=result.confidence,
            features=result.features
        )
    except Exception as e:
        logger.error(f"Error in customer segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing customer segmentation")

@router.post("/batch-segment", response_model=List[SegmentationResponse])
async def batch_segment_customers(
    request: BatchSegmentationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Batch customer segmentation requested by user {current_user.id}")
    service = CustomerSegmentationService(db)
    try:
        results = await asyncio.gather(*[service.segment_customer(customer_id) for customer_id in request.customer_ids])
        background_tasks.add_task(service.store_batch_results, results, current_user.id)
        logger.info(f"Batch customer segmentation completed for user {current_user.id}")
        return [
            SegmentationResponse(
                customer_id=customer_id,
                segment=result.segment,
                confidence=result.confidence,
                features=result.features
            )
            for result, customer_id in zip(results, request.customer_ids)
        ]
    except Exception as e:
        logger.error(f"Error in batch customer segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing batch customer segmentation")

@router.get("/models", response_model=List[SegmentationModel])
async def get_segmentation_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Segmentation models requested by user {current_user.id}")
    service = CustomerSegmentationService(db)
    try:
        models = await service.get_available_models()
        return [SegmentationModel(id=model.id, name=model.name, description=model.description) for model in models]
    except Exception as e:
        logger.error(f"Error retrieving segmentation models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving segmentation models")

@router.post("/update-segment", response_model=SegmentationResponse)
async def update_customer_segment(
    request: SegmentUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Customer segment update requested by user {current_user.id}")
    service = CustomerSegmentationService(db)
    try:
        result = await service.update_customer_segment(request.customer_id, request.new_segment)
        logger.info(f"Customer segment updated for user {current_user.id}")
        return SegmentationResponse(
            customer_id=request.customer_id,
            segment=result.segment,
            confidence=1.0,  # Manual update, so confidence is 100%
            features=result.features
        )
    except Exception as e:
        logger.error(f"Error updating customer segment: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating customer segment")

@router.get("/segment-distribution", response_model=dict)
async def get_segment_distribution(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Segment distribution requested by user {current_user.id}")
    service = CustomerSegmentationService(db)
    try:
        distribution = await service.get_segment_distribution()
        return distribution
    except Exception as e:
        logger.error(f"Error retrieving segment distribution: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving segment distribution")
