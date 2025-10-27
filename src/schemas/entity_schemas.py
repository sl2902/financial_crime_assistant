"""Pydantic schemas for extracted entities for SEC financial crime documents"""

from datetime import datetime, date
from enum import Enum
from typing import List, Optional
from loguru import logger

from pydantic import BaseModel, Field, field_validator

class EntityType(str, Enum):
    """Types of entities in financial crime cases"""

    PERSON = "Person"
    COMPANY = "Company"
    CASE = "Case"
    PENALTY = "Penalty"

class CrimeType(str, Enum):
    """Types of financial crimes - matches Qdrant crime type classification."""

    PONZI_SCHEME = "Ponzi Scheme"
    INSIDER_TRADING = "Insider Trading"
    SECURITIES_FRAUD = "Securities Fraud"
    WIRE_FRAUD = "Wire Fraud"
    MONEY_LAUNDERING = "Money Laundering"
    EMBEZZLEMENT = "Embezzlement"
    MARKET_MANIPULATION = "Market Manipulation"
    BRIBERY_FCPA = "Bribery/FCPA"
    FRAUD_GENERAL = "Fraud (General)"
    FINANCIAL_CRIME_OTHER = "Financial Crime (Other)"

class PenaltyType(str, Enum):
    """Types of penalties"""

    MONETARY = "Monetary"
    INJUNCTION = "Injunction"
    SUSPENSION = "Suspension"
    BAR = "Bar"
    DISGORGEMENT = "Disgorgement"
    CEASE_AND_DESIST = "Cease and Desist"
    OTHER = "Other"

class Person(BaseModel):
    """Represents an individual involved in a financial crime case"""

    name: str = Field(..., description="Full name of the person")
    role: Optional[str] = Field(None, description="Role/title (e.g., CEO, CFO, Trader)")
    company_affiliations: List[str] = Field(
            default_factory=list, description="Companies associated with this person"
    )

    @field_validator("name")
    @classmethod
    def clean_names(cls, v: str) -> str:
        """Clean and normalize person names"""
        return " ".join(v.strip().split())

class Company(BaseModel):
    """Represents a company involved in a financial crime case"""

    name: str = Field(..., description="Company name")
    ticker: Optional[str] = Field(None, description="Stock ticker symbol if available")
    industry: Optional[str] = Field(None, description="Industry/sector")
    description: Optional[str] = Field(None, description="Brief description")

    @field_validator("name")
    @classmethod
    def clean_company_name(cls, v: str) -> str:
        """Clean and normalize company names"""
        return " ".join(v.strip().split())

class LRNumberMixin:
    """Mixin for models that need LR number validation"""
    
    lr_number: str = Field(..., description="Litigation Release number (e.g., LR-12345)")
    
    @field_validator("lr_number")
    @classmethod
    def validate_lr_number(cls, v: str) -> str:
        """Ensure LR number is properly formatted as LR-XXXXX."""
        v = v.strip().upper()
        
        if v.startswith("LR-") and v[3:].isdigit():
            return v
        if v.startswith("LR") and v[2:].isdigit():
            return f"LR-{v[2:]}"
        if v.isdigit():
            return f"LR-{v}"
        
        raise ValueError(
            f"Invalid LR number format: '{v}'. "
            f"Expected format: LR-XXXXX (e.g., LR-25000)"
        )

class Case(LRNumberMixin, BaseModel):
    """Represents an SEC enforcement case"""

    # lr_number: str = Field(..., description="Litigation Release number (e.g., LR-12345)")
    title: str = Field(..., description="Case title")
    crime_types: List[CrimeType] = Field(
        default_factory=list, description="Types of crimes involved"
    )
    url: str = Field(..., description="SEC document URL")
    filing_date: date = Field(..., description="Date case was filed")
    summary: Optional[str] = Field(None, description="Brief case summary")

    @field_validator("filing_date", mode="before")
    @classmethod
    def parse_filing_date(cls, v):
        """Parse various date formats to date object."""
        if isinstance(v, date):
            return v
        
        if isinstance(v, str):
            v = v.strip()
            
            # Try common formats
            formats = [
                "%Y-%m-%d",           # 2025-01-15
                "%m/%d/%Y",           # 01/15/2025
                "%B %d, %Y",          # January 15, 2025
                "%b %d, %Y",          # Jan 15, 2025
                "%d-%m-%Y",           # 15-01-2025
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(v, fmt).date()
                except ValueError:
                    continue
            
            raise ValueError(f"Invalid date format: '{v}'. Expected YYYY-MM-DD")
        
        raise ValueError(f"Invalid date type: {type(v)}")

    # @field_validator("lr_number")
    # @classmethod
    # def validate_lr_number(cls, v: str) -> str:
    #     """Ensure LR number is properly formatted"""
    #     v = v.strip().upper()
    #     if v.startswith("LR-") and v[3:].isdigit():
    #         return v
    
    #     # Case 2: Missing hyphen "LR12345" -> "LR-12345"
    #     if v.startswith("LR") and v[2:].isdigit():
    #             return f"LR-{v[2:]}"
        
    #     # Case 3: Just the number "12345" -> "LR-12345"
    #     if v.isdigit():
    #             return f"LR-{v}"
        
    #     # Case 4: Invalid format - raise error
    #     raise ValueError(
    #         f"Invalid LR number format: '{v}'. "
    #         f"Expected format: LR-XXXXX (e.g., LR-25000)"
    #     )

class Penalty(BaseModel):
    """Represents a penalty imposed in a case"""
    
    penalty_type: PenaltyType = Field(..., description="Type of penalty")
    amount: Optional[float] = Field(None, description="Monetary amount (if applicable)")
    description: str = Field(..., description="Description of the penalty")
    recipient: Optional[str] = Field(
        None, description="Person or company receiving the penalty"
    )

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: Optional[float]) -> Optional[float]:
        """Ensure penalty amount is positive."""
        if v is not None and v < 0:
            raise ValueError("Penalty amount must be positive")
        return v

class ExtractedEntities(LRNumberMixin, BaseModel):
    """Container for all entities extracted from a document"""

    persons: List[Person] = Field(default_factory=list, description="Extracted persons")
    companies: List[Company] = Field(default_factory=list, description="Extracted companies")
    case: Case = Field(..., description="Case information")
    penalties: List[Penalty] = Field(default_factory=list, description="Extracted penalties")
    raw_text: Optional[str] = Field(None, description="Original document text")

    def summary_stats(self) -> dict:
        """Return summary statistics of extracted entities"""
        return {
            "lr_number": self.lr_number,
            "num_persons": len(self.persons),
            "num_companies": len(self.companies),
            "num_penalties": len(self.penalties),
            "crime_types": [ct.value for ct in self.case.crime_types],
        }


if __name__ == "__main__":
    # Test entity creation
    person = Person(
        name="John Doe", role="CEO", company_affiliations=["Acme Corp", "TechStart Inc"]
    )
    logger.info(f"Person: {person.name} - {person.role}")

    company = Company(name="Acme Corp", ticker="ACME", industry="Technology")
    logger.info(f"Company: {company.name} ({company.ticker})")

    case = Case(
        lr_number="12345",
        url="https://www.sec.gov/enforcement-litigation/litigation-releases/lr-26155",
        title="SEC v. Acme Corp",
        filing_date="2024-10-24",
        crime_types=[CrimeType.SECURITIES_FRAUD],
    )
    logger.info(f"Case: {case.lr_number} - {case.title} - {case.filing_date}")

    penalty = Penalty(
        penalty_type=PenaltyType.MONETARY,
        amount=1000000.0,
        description="Civil penalty of $1 million",
        recipient="John Doe",
    )
    logger.info(f"Penalty: {penalty.penalty_type.value} - ${penalty.amount:,.0f}")

