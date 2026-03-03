import os
import logging

class CloudAdapter:
    """
    Standardized adapter for cloud-agnostic operations.
    """
    
    @staticmethod
    def get_secret(secret_id: str) -> str:
        """
        Retrieves a secret from Google Cloud Secret Manager.
        Falls back to local environment variables if GCP fails or is not configured.
        """
        try:
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set.")
                
            name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
            
        except ImportError:
            logging.warning("google-cloud-secret-manager not installed. Falling back to env vars.")
            return os.environ.get(secret_id, f"mock-secret-{secret_id}")
            
        except Exception as e:
            logging.warning(f"Failed to fetch secret from GCP: {e}. Falling back to env vars.")
            return os.environ.get(secret_id, f"mock-secret-{secret_id}")
