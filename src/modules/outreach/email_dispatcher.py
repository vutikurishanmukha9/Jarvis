"""
Email Dispatcher for Smart HR Outreach & Cold Email Engine.
Supports secure SMTP TLS/SSL delivery and simulated dry-run campaign previews with audit logging.

SECURITY: Live SMTP dispatch requires explicit human approval through the UI.
The autonomous agent is restricted to simulation mode only.
"""

import time
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any, Optional
import pandas as pd

from ...config import WORKSPACE_DIR
from .campaign_manager import CampaignManager

logger = logging.getLogger(__name__)

class EmailDispatcher:
    """Dispatches cold outreach campaigns via live SMTP or sandboxed simulation."""

    @staticmethod
    def dispatch(
        subject_template: str,
        body_template: str,
        recipients: List[Dict[str, Any]],
        global_tags: Optional[Dict[str, Any]] = None,
        smtp_config: Optional[Dict[str, Any]] = None,
        simulated: bool = True,
        delay_seconds: float = 0.05
    ) -> Dict[str, Any]:
        """
        Execute campaign delivery for a list of recipients.
        
        Args:
            subject_template: Subject with dynamic {tag} variables.
            body_template: Body with dynamic {tag} variables.
            recipients: List of recipient dicts.
            global_tags: Additional global variables (e.g. candidateName, portfolio).
            smtp_config: Dict with host, port, user, password, from_email.
            simulated: If True, executes in safe sandbox mode without sending live network emails.
            delay_seconds: Throttle delay between emails.
        """
        if not recipients:
            return {
                "status": "error",
                "message": "No recipients provided for dispatch.",
                "total": 0,
                "sent": 0,
                "failed": 0
            }

        sent_count = 0
        failed_count = 0
        delivery_logs: List[Dict[str, Any]] = []

        server = None
        if not simulated and smtp_config:
            try:
                host = smtp_config.get("host", "smtp.gmail.com")
                port = int(smtp_config.get("port", 587))
                user = smtp_config.get("user", "")
                pwd = smtp_config.get("password", "")
                
                server = smtplib.SMTP(host, port, timeout=15)
                server.starttls()
                if user and pwd:
                    server.login(user, pwd)
            except Exception as e:
                logger.error(f"Failed to connect to SMTP server: {e}")
                return {
                    "status": "error",
                    "message": f"SMTP Connection Failed: {str(e)}",
                    "total": len(recipients),
                    "sent": 0,
                    "failed": len(recipients)
                }

        campaign_id = f"camp_{int(time.time())}"
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")

        for idx, rec in enumerate(recipients):
            email_addr = rec.get("email", "").strip()
            if not email_addr or "@" not in email_addr:
                failed_count += 1
                delivery_logs.append({
                    "recipient": email_addr or f"Row {idx+1}",
                    "name": rec.get("firstName", "Unknown"),
                    "company": rec.get("company", "Unknown"),
                    "status": "Failed (Invalid Email)",
                    "timestamp": time.strftime("%H:%M:%S")
                })
                continue

            rendered_subj = CampaignManager.render_template(subject_template, rec, global_tags)
            rendered_body = CampaignManager.render_template(body_template, rec, global_tags)

            if simulated:
                # Sandboxed execution simulation
                time.sleep(delay_seconds)
                sent_count += 1
                delivery_logs.append({
                    "recipient": email_addr,
                    "name": rec.get("firstName", "there"),
                    "company": rec.get("company", "your organization"),
                    "status": "Simulated Sent",
                    "subject": rendered_subj,
                    "timestamp": time.strftime("%H:%M:%S")
                })
            else:
                # Live SMTP sending
                try:
                    from_email = smtp_config.get("from_email", smtp_config.get("user", "outreach@jarvis.ai"))
                    msg = MIMEMultipart()
                    msg["From"] = from_email
                    msg["To"] = email_addr
                    msg["Subject"] = rendered_subj
                    msg.attach(MIMEText(rendered_body, "plain"))

                    if server:
                        server.sendmail(from_email, [email_addr], msg.as_string())
                    
                    sent_count += 1
                    delivery_logs.append({
                        "recipient": email_addr,
                        "name": rec.get("firstName", "there"),
                        "company": rec.get("company", "your organization"),
                        "status": "Delivered",
                        "subject": rendered_subj,
                        "timestamp": time.strftime("%H:%M:%S")
                    })
                    time.sleep(delay_seconds)
                except Exception as ex:
                    failed_count += 1
                    delivery_logs.append({
                        "recipient": email_addr,
                        "name": rec.get("firstName", "there"),
                        "company": rec.get("company", "your organization"),
                        "status": f"Failed: {str(ex)[:50]}",
                        "timestamp": time.strftime("%H:%M:%S")
                    })

        if server:
            try:
                server.quit()
            except Exception:
                pass

        # Generate Excel Audit Sheet in workspace
        audit_filename = f"outreach_audit_{campaign_id}.xlsx"
        audit_path = WORKSPACE_DIR / audit_filename
        try:
            df_audit = pd.DataFrame(delivery_logs)
            df_audit.to_excel(audit_path, index=False)
        except Exception as e:
            logger.warning(f"Could not write audit spreadsheet: {e}")

        # Save Campaign Record
        record = {
            "campaign_id": campaign_id,
            "timestamp": start_time,
            "simulated": simulated,
            "subject": subject_template,
            "total_recipients": len(recipients),
            "sent": sent_count,
            "failed": failed_count,
            "audit_file": audit_filename if audit_path.exists() else None
        }
        CampaignManager.save_campaign_record(record)

        return {
            "status": "success",
            "campaign_id": campaign_id,
            "simulated": simulated,
            "total": len(recipients),
            "sent": sent_count,
            "failed": failed_count,
            "delivery_logs": delivery_logs,
            "audit_file": str(audit_path) if audit_path.exists() else None,
            "message": f"Campaign {'simulation' if simulated else 'delivery'} complete: {sent_count}/{len(recipients)} sent successfully."
        }
