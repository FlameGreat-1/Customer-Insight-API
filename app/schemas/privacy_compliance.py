from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

class RequestType(str, Enum):
    ACCESS = "access"
    RECTIFICATION = "rectification"
    ERASURE = "erasure"

class DataSubjectRequest(BaseModel):
    user_id: int = Field(..., description="The ID of the user making the request")
    request_type: RequestType = Field(..., description="The type of data subject request")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data for the request")

class DataSubjectResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    user_id: int = Field(..., description="The ID of the user who made the request")
    request_type: RequestType = Field(..., description="The type of data subject request")
    status: str = Field(..., description="The status of the request")
    completion_date: datetime = Field(..., description="The date when the request was or will be completed")
    data: Optional[Dict[str, Any]] = Field(None, description="Data returned for access requests")

class ConsentManagementRequest(BaseModel):
    user_id: int = Field(..., description="The ID of the user managing consent")
    consents: Dict[str, bool] = Field(..., description="Dictionary of consent types and their status")

class ConsentManagementResponse(BaseModel):
    user_id: int = Field(..., description="The ID of the user")
    consents: Dict[str, bool] = Field(..., description="Updated dictionary of consent types and their status")
    last_updated: datetime = Field(..., description="Timestamp of when the consents were last updated")

class DataRetentionPolicyRequest(BaseModel):
    data_type: str = Field(..., description="The type of data for which the retention policy is being set")
    retention_period: int = Field(..., description="The retention period in days")

class DataRetentionPolicyResponse(BaseModel):
    policy_id: str = Field(..., description="Unique identifier for the policy")
    data_type: str = Field(..., description="The type of data for which the retention policy is set")
    retention_period: int = Field(..., description="The retention period in days")
    last_updated: datetime = Field(..., description="Timestamp of when the policy was last updated")

class DataBreachNotificationRequest(BaseModel):
    breach_date: datetime = Field(..., description="The date when the data breach occurred")
    description: str = Field(..., description="Description of the data breach")
    affected_data_types: List[str] = Field(..., description="Types of data affected by the breach")

class DataBreachNotificationResponse(BaseModel):
    notification_id: str = Field(..., description="Unique identifier for the notification")
    breach_date: datetime = Field(..., description="The date when the data breach occurred")
    affected_users: int = Field(..., description="Number of users affected by the breach")
    notification_status: str = Field(..., description="Status of the notification process")

class AuditType(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"

class ComplianceAuditRequest(BaseModel):
    audit_type: AuditType = Field(..., description="Type of compliance audit")

class ComplianceAuditResponse(BaseModel):
    audit_id: str = Field(..., description="Unique identifier for the audit")
    audit_type: AuditType = Field(..., description="Type of compliance audit")
    start_date: datetime = Field(..., description="Start date of the audit")
    estimated_completion_date: datetime = Field(..., description="Estimated completion date of the audit")
    status: str = Field(..., description="Current status of the audit")
    findings: Optional[Dict[str, Any]] = Field(None, description="Audit findings")

class AnonymizationRequest(BaseModel):
    data_type: str = Field(..., description="The type of data to be anonymized")

class AnonymizationResponse(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the anonymization job")
    status: str = Field(..., description="Status of the anonymization job")

class ComplianceReportRequest(BaseModel):
    start_date: str = Field(..., description="Start date for the compliance report period")
    end_date: str = Field(..., description="End date for the compliance report period")

class ComplianceReportResponse(BaseModel):
    period: str = Field(..., description="Period covered by the report")
    data_subject_requests: int = Field(..., description="Number of data subject requests processed")
    consent_changes: int = Field(..., description="Number of consent changes recorded")
    data_breaches: int = Field(..., description="Number of data breaches reported")
    compliance_audits: int = Field(..., description="Number of compliance audits conducted")

class DataProcessingAgreementRequest(BaseModel):
    partner_id: str = Field(..., description="ID of the partner for which the agreement is being created")
    agreement_details: Dict[str, Any] = Field(..., description="Details of the data processing agreement")

class DataProcessingAgreementResponse(BaseModel):
    agreement_id: str = Field(..., description="Unique identifier for the agreement")
    partner_id: str = Field(..., description="ID of the partner")
    created_at: datetime = Field(..., description="Timestamp of when the agreement was created")

class PrivacyImpactAssessmentRequest(BaseModel):
    project_id: str = Field(..., description="ID of the project for which the assessment is being conducted")

class PrivacyImpactAssessmentResponse(BaseModel):
    project_id: str = Field(..., description="ID of the project")
    risk_areas: List[str] = Field(..., description="Areas of potential privacy risks")
    risk_levels: Dict[str, str] = Field(..., description="Risk levels for each risk area")
    recommendations: List[str] = Field(..., description="Recommendations to mitigate privacy risks")
