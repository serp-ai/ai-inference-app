"""
Authentication utilities for API key verification
"""

import os
from typing import Optional

from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


# API Key configuration
API_KEY = os.getenv("API_KEY")
security = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> bool:
    """
    Verify API key if authentication is enabled.
    
    Args:
        credentials: HTTP Bearer credentials from request header
        
    Returns:
        bool: True if authentication passes or is disabled
        
    Raises:
        HTTPException: 401 if API key is invalid or missing when required
    """
    if not API_KEY:
        return True  # No API key required
    
    if not credentials or credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


def is_auth_enabled() -> bool:
    """Check if API key authentication is enabled"""
    return API_KEY is not None