import asyncio
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.user import User
from app.models.consent import Consent
from app.models.data_retention_policy import DataRetentionPolicy
from app.models.data_breach_notification import DataBreachNotification
from app.models.compliance_audit import ComplianceAudit
from app.models.data_subject_request import DataSubjectRequest as DSR
from app.models.order import Order
from app.models.interaction import Interaction
from app.models.project import Project
from app.models.privacy_event import PrivacyEvent
from app.models.data_processing_agreement import DataProcessingAgreement
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.core.config import settings
from app.models.notification_log import NotificationLog
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
from app.core.logging import logger
from datetime import datetime, timedelta
import uuid

class PrivacyComplianceService:
    def __init__(self, db: Session):
        self.db = db

    async def process_data_subject_request(self, request: DataSubjectRequest) -> DataSubjectResponse:
        try:
            user = await self.db.query(User).filter(User.id == request.user_id).first()
            if not user:
                raise ValueError(f"User with id {request.user_id} not found")

            request_id = str(uuid.uuid4())
            completion_date = datetime.utcnow() + timedelta(days=30)

            result = await self.perform_data_subject_request(request)

            await self.log_privacy_event("Data Subject Request", f"Processed {request.request_type} request for user {request.user_id}")

            return DataSubjectResponse(
                request_id=request_id,
                user_id=user.id,
                request_type=request.request_type,
                status="Completed",
                completion_date=completion_date,
                data=result
            )
        except Exception as e:
            logger.error(f"Error processing data subject request: {str(e)}")
            raise

    async def manage_consent(self, request: ConsentManagementRequest) -> ConsentManagementResponse:
        try:
            user = await self.db.query(User).filter(User.id == request.user_id).first()
            if not user:
                raise ValueError(f"User with id {request.user_id} not found")

            for consent_type, consent_given in request.consents.items():
                consent = await self.db.query(Consent).filter(
                    Consent.user_id == user.id,
                    Consent.consent_type == consent_type
                ).first()

                if consent:
                    consent.consent_given = consent_given
                    consent.last_updated = datetime.utcnow()
                else:
                    new_consent = Consent(
                        user_id=user.id,
                        consent_type=consent_type,
                        consent_given=consent_given,
                        last_updated=datetime.utcnow()
                    )
                    self.db.add(new_consent)

            await self.db.commit()
            await self.log_privacy_event("Consent Management", f"Updated consent for user {request.user_id}")

            return ConsentManagementResponse(
                user_id=user.id,
                consents=request.consents,
                last_updated=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Error managing consent: {str(e)}")
            raise

    async def set_data_retention_policy(self, request: DataRetentionPolicyRequest) -> DataRetentionPolicyResponse:
        try:
            policy = await self.db.query(DataRetentionPolicy).filter(
                DataRetentionPolicy.data_type == request.data_type
            ).first()

            if policy:
                policy.retention_period = request.retention_period
                policy.last_updated = datetime.utcnow()
            else:
                policy = DataRetentionPolicy(
                    data_type=request.data_type,
                    retention_period=request.retention_period,
                    last_updated=datetime.utcnow()
                )
                self.db.add(policy)

            await self.db.commit()
            await self.log_privacy_event("Data Retention Policy", f"Updated policy for {request.data_type}")

            return DataRetentionPolicyResponse(
                policy_id=str(policy.id),
                data_type=policy.data_type,
                retention_period=policy.retention_period,
                last_updated=policy.last_updated
            )
        except Exception as e:
            logger.error(f"Error setting data retention policy: {str(e)}")
            raise

    async def initiate_data_breach_notification(self, request: DataBreachNotificationRequest) -> DataBreachNotificationResponse:
        try:
            notification = DataBreachNotification(
                breach_date=request.breach_date,
                description=request.description,
                affected_data_types=request.affected_data_types,
                notification_status="Initiated"
            )
            self.db.add(notification)
            await self.db.commit()

            affected_users = await self._get_affected_users(request.affected_data_types)
            asyncio.create_task(self.handle_data_breach(notification))

            await self.log_privacy_event("Data Breach", f"Initiated notification for breach on {request.breach_date}")

            return DataBreachNotificationResponse(
                notification_id=str(notification.id),
                breach_date=notification.breach_date,
                affected_users=len(affected_users),
                notification_status=notification.notification_status
            )
        except Exception as e:
            logger.error(f"Error initiating data breach notification: {str(e)}")
            raise

    async def initiate_compliance_audit(self, request: ComplianceAuditRequest) -> ComplianceAuditResponse:
        try:
            audit = ComplianceAudit(
                audit_type=request.audit_type,
                start_date=datetime.utcnow(),
                estimated_completion_date=datetime.utcnow() + timedelta(days=30),
                status="Initiated"
            )
            self.db.add(audit)
            await self.db.commit()

            asyncio.create_task(self._conduct_audit(audit.id))
            await self.log_privacy_event("Compliance Audit", f"Initiated {request.audit_type} audit")

            return ComplianceAuditResponse(
                audit_id=str(audit.id),
                audit_type=audit.audit_type,
                start_date=audit.start_date,
                estimated_completion_date=audit.estimated_completion_date,
                status=audit.status
            )
        except Exception as e:
            logger.error(f"Error initiating compliance audit: {str(e)}")
            raise

    async def get_audit_status(self, audit_id: str) -> ComplianceAuditResponse:
        try:
            audit = await self.db.query(ComplianceAudit).filter(ComplianceAudit.id == audit_id).first()
            if not audit:
                raise ValueError(f"Audit with id {audit_id} not found")

            return ComplianceAuditResponse(
                audit_id=str(audit.id),
                audit_type=audit.audit_type,
                start_date=audit.start_date,
                estimated_completion_date=audit.estimated_completion_date,
                status=audit.status,
                findings=audit.findings
            )
        except Exception as e:
            logger.error(f"Error retrieving audit status: {str(e)}")
            raise

    async def initiate_data_anonymization(self, data_type: str) -> str:
        try:
            job_id = str(uuid.uuid4())
            asyncio.create_task(self._anonymize_data(data_type, job_id))
            await self.log_privacy_event("Data Anonymization", f"Initiated anonymization for {data_type}")
            return job_id
        except Exception as e:
            logger.error(f"Error initiating data anonymization: {str(e)}")
            raise

    async def generate_compliance_report(self, start_date: str, end_date: str) -> Dict[str, Any]:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            report = {
                "period": f"{start_date} to {end_date}",
                "data_subject_requests": await self._count_data_subject_requests(start, end),
                "consent_changes": await self._count_consent_changes(start, end),
                "data_breaches": await self._count_data_breaches(start, end),
                "compliance_audits": await self._count_compliance_audits(start, end)
            }
            await self.log_privacy_event("Compliance Report", f"Generated report for period {start_date} to {end_date}")
            return report
        except Exception as e:
            logger.error(f"Error generating compliance report: {str(e)}")
            raise

    async def create_data_processing_agreement(self, partner_id: str, agreement_details: Dict[str, Any]) -> Dict[str, Any]:
        try:
            agreement_id = str(uuid.uuid4())
            agreement = DataProcessingAgreement(
                id=agreement_id,
                partner_id=partner_id,
                details=agreement_details,
                created_at=datetime.utcnow()
            )
            self.db.add(agreement)
            await self.db.commit()
            await self.log_privacy_event("Data Processing Agreement", f"Created agreement with partner {partner_id}")
            return {
                "id": agreement_id,
                "partner_id": partner_id,
                "details": agreement_details,
                "created_at": agreement.created_at
            }
        except Exception as e:
            logger.error(f"Error creating data processing agreement: {str(e)}")
            raise

    async def conduct_privacy_impact_assessment(self, project_id: str) -> Dict[str, Any]:
        try:
            project = await self.db.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise ValueError(f"Project with id {project_id} not found")

            assessment = {
                "project_id": project_id,
                "project_name": project.name,
                "risk_areas": ["data collection", "data storage", "data processing", "data sharing"],
                "risk_levels": await self._assess_risk_levels(project),
                "recommendations": await self._generate_recommendations(project)
            }
            await self.log_privacy_event("Privacy Impact Assessment", f"Conducted assessment for project {project_id}")
            return assessment
        except Exception as e:
            logger.error(f"Error conducting privacy impact assessment: {str(e)}")
            raise

    # Helper methods
    async def _get_user_data(self, user_id: int) -> Dict[str, Any]:
        user = await self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User with id {user_id} not found")
        return {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "created_at": user.created_at,
        }

    async def _update_user_data(self, user_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        user = await self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User with id {user_id} not found")
        for key, value in data.items():
            setattr(user, key, value)
        await self.db.commit()
        return await self._get_user_data(user_id)

    async def _delete_user_data(self, user_id: int) -> Dict[str, Any]:
        user = await self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User with id {user_id} not found")
        deleted_data = await self._get_user_data(user_id)
        await self.db.delete(user)
        await self.db.commit()
        return deleted_data

    async def _get_affected_users(self, affected_data_types: List[str]) -> List[User]:
        return await self.db.query(User).filter(User.data_types.overlap(affected_data_types)).all()

    async def _count_data_subject_requests(self, start_date: datetime, end_date: datetime) -> int:
        return await self.db.query(func.count(DSR.id)).filter(
            DSR.created_at.between(start_date, end_date)
        ).scalar()

    async def _count_consent_changes(self, start_date: datetime, end_date: datetime) -> int:
        return await self.db.query(func.count(Consent.id)).filter(
            Consent.last_updated.between(start_date, end_date)
        ).scalar()

    async def _count_data_breaches(self, start_date: datetime, end_date: datetime) -> int:
        return await self.db.query(func.count(DataBreachNotification.id)).filter(
            DataBreachNotification.breach_date.between(start_date, end_date)
        ).scalar()

    async def _count_compliance_audits(self, start_date: datetime, end_date: datetime) -> int:
        return await self.db.query(func.count(ComplianceAudit.id)).filter(
            ComplianceAudit.start_date.between(start_date, end_date)
        ).scalar()

    async def _anonymize_data(self, data_type: str, job_id: str):
        try:
            users = await self.db.query(User).filter(User.data_types.contains([data_type])).all()
            for user in users:
                if data_type == "personal_info":
                    user.name = f"Anonymous User {user.id}"
                    user.email = f"anonymous{user.id}@example.com"
                elif data_type == "location_data":
                    user.address = "Anonymized"
                    user.city = "Anonymized"
                    user.country = "Anonymized"
            
            await self.db.commit()
            await self.log_privacy_event("Data Anonymization", f"Completed anonymization for {data_type}")
            logger.info(f"Data anonymization completed for data type: {data_type}, job ID: {job_id}")
        except Exception as e:
            logger.error(f"Error in data anonymization job: {str(e)}")

    async def _assess_risk_levels(self, project: Project) -> Dict[str, str]:
        risk_levels = {}
        
        risk_levels["data collection"] = "low" if project.data_collection_minimal else "high"
        risk_levels["data storage"] = "medium"
        risk_levels["data processing"] = "low" if project.data_processing_documented else "high"
        risk_levels["data sharing"] = "high" if project.data_sharing_enabled else "low"
        
        return risk_levels

    async def _generate_recommendations(self, project: Project) -> List[str]:
        recommendations = []
        
        if not project.data_collection_minimal:
            recommendations.append("Minimize data collection to only necessary information")
        if not project.data_processing_documented:
            recommendations.append("Document all data processing activities")
        if project.data_sharing_enabled:
            recommendations.append("Review and limit data sharing practices")
        recommendations.append("Implement encryption for data storage")
        
        return recommendations

    async def perform_data_subject_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        user = await self.db.query(User).filter(User.id == request.user_id).first()
        if not user:
            raise ValueError(f"User with id {request.user_id} not found")

        if request.request_type == "access":
            return await self._get_user_data(user.id)
        elif request.request_type == "rectification":
            return await self._update_user_data(user.id, request.data)
        elif request.request_type == "erasure":
            return await self._delete_user_data(user.id)
        elif request.request_type == "restrict_processing":
            user.data_processing_restricted = True
            await self.db.commit()
            return {"message": "Data processing restricted successfully"}
        elif request.request_type == "data_portability":
            return await self._export_user_data(user.id)
        else:
            raise ValueError(f"Invalid request type: {request.request_type}")

    async def _export_user_data(self, user_id: int) -> Dict[str, Any]:
        user_data = await self._get_user_data(user_id)
        user_data["orders"] = await self._get_user_orders(user_id)
        user_data["interactions"] = await self._get_user_interactions(user_id)
        return user_data

    async def _get_user_orders(self, user_id: int) -> List[Dict[str, Any]]:
        orders = await self.db.query(Order).filter(Order.user_id == user_id).all()
        return [{"id": order.id, "date": order.date, "total": order.total} for order in orders]

    async def _get_user_interactions(self, user_id: int) -> List[Dict[str, Any]]:
        interactions = await self.db.query(Interaction).filter(Interaction.user_id == user_id).all()
        return [{"id": interaction.id, "type": interaction.type, "date": interaction.date} for interaction in interactions]

    async def handle_data_breach(self, breach: DataBreachNotification) -> None:
        logger.critical(f"Data breach occurred: {breach.description}")

        affected_users = await self._get_affected_users(breach.affected_data_types)
        for user in affected_users:
            await self._notify_user_of_breach(user, breach)

        breach.notification_status = "Completed"
        await self.db.commit()
        await self.log_privacy_event("Data Breach", f"Completed notification for breach {breach.id}")


    async def _notify_user_of_breach(self, user: User, breach: DataBreachNotification) -> None:
        logger.info(f"Notifying user {user.id} of data breach: {breach.id}")
        
        try:
            # Prepare email content
            subject = "Important: Data Breach Notification"
            body = f"""
            Dear {user.name},

            We regret to inform you that a data breach occurred on {breach.breach_date} that may have affected your personal information.

            The breach involved the following types of data: {', '.join(breach.affected_data_types)}

            Description of the breach: {breach.description}

            We are taking this matter very seriously and have taken the following steps:
            1. We have contained the breach and are conducting a thorough investigation.
            2. We have reported the incident to the relevant authorities.
            3. We are reviewing and strengthening our security measures to prevent future incidents.

            What you can do:
            1. Be vigilant about any suspicious activities related to your personal information.
            2. Consider changing your passwords for our service and any other services where you might have used the same password.
            3. Monitor your financial statements and credit reports for any unauthorized activities.

            If you have any questions or concerns, please don't hesitate to contact our support team at support@OmniLens.com.

            We sincerely apologize for any inconvenience or concern this may cause you.

            Best regards,
            The Security Team
            """

            # Create the email message
            msg = MIMEMultipart()
            msg['From'] = settings.SMTP_SENDER
            msg['To'] = user.email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            # Send the email
            with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:
                server.starttls()
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                server.send_message(msg)

            # Log the notification
            notification_log = NotificationLog(
                user_id=user.id,
                breach_id=breach.id,
                notification_type="email",
                sent_at=datetime.utcnow()
            )
            self.db.add(notification_log)
            await self.db.commit()

            # Update user's record to indicate they've been notified
            user.last_breach_notification = datetime.utcnow()
            await self.db.commit()

            logger.info(f"Successfully notified user {user.id} of data breach: {breach.id}")

        except Exception as e:
            logger.error(f"Failed to notify user {user.id} of data breach: {breach.id}. Error: {str(e)}")
            raise



    async def generate_privacy_policy(self) -> str:
        policy = """
        Privacy Policy

        1. Data Collection: We collect personal information necessary for providing our services.
        2. Data Use: Your data is used to improve our services and personalize your experience.
        3. Data Sharing: We do not sell your personal data. We may share data with service providers under strict confidentiality agreements.
        4. Data Security: We implement robust security measures to protect your data.
        5. Your Rights: You have the right to access, rectify, erase, and port your personal data.
        6. Contact Us: For any privacy-related queries, contact our Data Protection Officer at OmniLens@.com.

        Last Updated: {date}
        """.format(date=datetime.utcnow().strftime("%Y-%m-%d"))

        await self.log_privacy_event("Privacy Policy", "Generated new privacy policy")
        return policy

    async def log_privacy_event(self, event_type: str, description: str) -> None:
        privacy_event = PrivacyEvent(
            event_type=event_type,
            description=description,
            timestamp=datetime.utcnow()
        )
        self.db.add(privacy_event)
        await self.db.commit()
        logger.info(f"Privacy event logged: {event_type} - {description}")

    async def _conduct_audit(self, audit_id: str) -> None:
        try:
            audit = await self.db.query(ComplianceAudit).filter(ComplianceAudit.id == audit_id).first()
            if not audit:
                raise ValueError(f"Audit with id {audit_id} not found")

            # Simulate audit process
            await asyncio.sleep(5)  # Simulating time-consuming audit process

            # Update audit status and findings
            audit.status = "Completed"
            audit.findings = {
                "data_collection": "Compliant",
                "data_processing": "Needs improvement",
                "data_sharing": "Compliant",
                "user_rights": "Compliant"
            }
            await self.db.commit()
            await self.log_privacy_event("Compliance Audit", f"Completed audit {audit_id}")
        except Exception as e:
            logger.error(f"Error conducting audit: {str(e)}")
            # Update audit status to failed
            if audit:
                audit.status = "Failed"
                await self.db.commit()
