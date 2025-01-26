from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.security import get_current_user
from app.schemas.privacy_compliance import (
    DataSubjectRequest,
    DataSubjectResponse,
    ConsentManagementRequest,
    ConsentManagementResponse,
    DataRetentionPolicyRequest,
    DataRetentionPolicyResponse,
    DataBreachNotificationRequest,
    DataBreachNotificationResponse,
    ComplianceAuditRequest,
    ComplianceAuditResponse
)
from app.services.privacy_compliance import PrivacyComplianceService
from app.models.user import User
from app.core.logging import logger
from typing import List
import asyncio

router = APIRouter()

@router.post("/data-subject-request", response_model=DataSubjectResponse)
async def handle_data_subject_request(
    request: DataSubjectRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Data subject request received for user {request.user_id}")
    service = PrivacyComplianceService(db)
    try:
        result = await service.process_data_subject_request(request)
        logger.info(f"Data subject request processed for user {request.user_id}")
        return DataSubjectResponse(
            request_id=result.request_id,
            user_id=request.user_id,
            request_type=request.request_type,
            status=result.status,
            completion_date=result.completion_date,
            data=result.data
        )
    except Exception as e:
        logger.error(f"Error processing data subject request: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing data subject request")

@router.post("/manage-consent", response_model=ConsentManagementResponse)
async def manage_consent(
    request: ConsentManagementRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Consent management request received for user {request.user_id}")
    service = PrivacyComplianceService(db)
    try:
        result = await service.manage_consent(request)
        logger.info(f"Consent updated for user {request.user_id}")
        return ConsentManagementResponse(
            user_id=request.user_id,
            consents=result.consents,
            last_updated=result.last_updated
        )
    except Exception as e:
        logger.error(f"Error managing consent: {str(e)}")
        raise HTTPException(status_code=500, detail="Error managing consent")

@router.post("/data-retention-policy", response_model=DataRetentionPolicyResponse)
async def set_data_retention_policy(
    request: DataRetentionPolicyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Data retention policy update requested by user {current_user.id}")
    service = PrivacyComplianceService(db)
    try:
        result = await service.set_data_retention_policy(request)
        logger.info(f"Data retention policy updated")
        return DataRetentionPolicyResponse(
            policy_id=result.policy_id,
            data_type=request.data_type,
            retention_period=request.retention_period,
            last_updated=result.last_updated
        )
    except Exception as e:
        logger.error(f"Error setting data retention policy: {str(e)}")
        raise HTTPException(status_code=500, detail="Error setting data retention policy")

@router.post("/data-breach-notification", response_model=DataBreachNotificationResponse)
async def notify_data_breach(
    request: DataBreachNotificationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Data breach notification initiated by user {current_user.id}")
    service = PrivacyComplianceService(db)
    try:
        result = await service.initiate_data_breach_notification(request)
        background_tasks.add_task(service.send_data_breach_notifications, result.notification_id)
        logger.info(f"Data breach notification process initiated")
        return DataBreachNotificationResponse(
            notification_id=result.notification_id,
            breach_date=request.breach_date,
            affected_users=result.affected_users,
            notification_status=result.notification_status
        )
    except Exception as e:
        logger.error(f"Error initiating data breach notification: {str(e)}")
        raise HTTPException(status_code=500, detail="Error initiating data breach notification")

@router.post("/compliance-audit", response_model=ComplianceAuditResponse)
async def conduct_compliance_audit(
    request: ComplianceAuditRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Compliance audit requested by user {current_user.id}")
    service = PrivacyComplianceService(db)
    try:
        result = await service.initiate_compliance_audit(request)
        background_tasks.add_task(service.perform_compliance_audit, result.audit_id)
        logger.info(f"Compliance audit initiated")
        return ComplianceAuditResponse(
            audit_id=result.audit_id,
            audit_type=request.audit_type,
            start_date=result.start_date,
            estimated_completion_date=result.estimated_completion_date,
            status=result.status
        )
    except Exception as e:
        logger.error(f"Error initiating compliance audit: {str(e)}")
        raise HTTPException(status_code=500, detail="Error initiating compliance audit")

@router.get("/audit-status/{audit_id}", response_model=ComplianceAuditResponse)
async def get_audit_status(
    audit_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Audit status requested for audit {audit_id} by user {current_user.id}")
    service = PrivacyComplianceService(db)
    try:
        result = await service.get_audit_status(audit_id)
        return ComplianceAuditResponse(
            audit_id=result.audit_id,
            audit_type=result.audit_type,
            start_date=result.start_date,
            estimated_completion_date=result.estimated_completion_date,
            status=result.status,
            findings=result.findings
        )
    except Exception as e:
        logger.error(f"Error retrieving audit status: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving audit status")

@router.post("/anonymize-data", response_model=dict)
async def anonymize_data(
    data_type: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Data anonymization requested for {data_type} by user {current_user.id}")
    service = PrivacyComplianceService(db)
    try:
        job_id = await service.initiate_data_anonymization(data_type)
        background_tasks.add_task(service.perform_data_anonymization, job_id)
        return {"status": "Data anonymization job initiated", "job_id": job_id}
    except Exception as e:
        logger.error(f"Error initiating data anonymization: {str(e)}")
        raise HTTPException(status_code=500, detail="Error initiating data anonymization")

@router.get("/compliance-report", response_model=dict)
async def generate_compliance_report(
    start_date: str,
    end_date: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Compliance report requested by user {current_user.id}")
    service = PrivacyComplianceService(db)
    try:
        report = await service.generate_compliance_report(start_date, end_date)
        return report
    except Exception as e:
        logger.error(f"Error generating compliance report: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating compliance report")

@router.post("/data-processing-agreement", response_model=dict)
async def create_data_processing_agreement(
    partner_id: str,
    agreement_details: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Data processing agreement creation requested for partner {partner_id} by user {current_user.id}")
    service = PrivacyComplianceService(db)
    try:
        agreement = await service.create_data_processing_agreement(partner_id, agreement_details)
        return {"status": "Data processing agreement created", "agreement_id": agreement.id}
    except Exception as e:
        logger.error(f"Error creating data processing agreement: {str(e)}")
        raise HTTPException(status_code=500, detail="Error creating data processing agreement")

@router.get("/privacy-impact-assessment", response_model=dict)
async def conduct_privacy_impact_assessment(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Privacy impact assessment requested for project {project_id} by user {current_user.id}")
    service = PrivacyComplianceService(db)
    try:
        assessment = await service.conduct_privacy_impact_assessment(project_id)
        return assessment
    except Exception as e:
        logger.error(f"Error conducting privacy impact assessment: {str(e)}")
        raise HTTPException(status_code=500, detail="Error conducting privacy impact assessment")
