from pydantic import BaseModel, Field
from typing import Optional


class Entity(BaseModel):
    """A single named entity extracted from text."""
    text: str = Field(description="The exact text of the entity")
    type: str = Field(description="Type: PERSON, ORG, LOCATION, DATE, or OTHER")
    confidence: str = Field(description="Confidence: high, medium, or low")


class ExtractionResult(BaseModel):
    """Full structured extraction from a piece of text."""
    persons: list[str] = Field(
        default=[],
        description="Full names of people mentioned"
    )
    organizations: list[str] = Field(
        default=[],
        description="Company, institution, or organization names"
    )
    locations: list[str] = Field(
        default=[],
        description="Cities, countries, addresses, or places"
    )
    dates: list[str] = Field(
        default=[],
        description="Dates, years, time periods mentioned"
    )
    key_facts: list[str] = Field(
        default=[],
        description="The 3 most important facts from the text"
    )
    sentiment: str = Field(
        default="neutral",
        description="Overall sentiment: positive, negative, or neutral"
    )
    summary: str = Field(
        default="",
        description="One sentence summary of the text"
    )


class ContactInfo(BaseModel):
    """Structured contact information extracted from text."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    location: Optional[str] = None


class MeetingNotes(BaseModel):
    """Structured data extracted from meeting notes."""
    title: str = Field(default="", description="Meeting title or topic")
    date: Optional[str] = Field(default=None, description="Date of the meeting")
    attendees: list[str] = Field(default=[], description="Names of people present")
    decisions: list[str] = Field(default=[], description="Decisions made")
    action_items: list[str] = Field(default=[], description="Tasks assigned")
    next_meeting: Optional[str] = Field(default=None, description="Next meeting date if mentioned")


class ProductReview(BaseModel):
    """Structured data extracted from a product review."""
    product_name: str = Field(default="", description="Name of the product")
    rating: Optional[int] = Field(default=None, description="Rating 1-5 if mentioned")
    pros: list[str] = Field(default=[], description="Positive aspects mentioned")
    cons: list[str] = Field(default=[], description="Negative aspects mentioned")
    sentiment: str = Field(default="neutral", description="positive, negative, or neutral")
    would_recommend: Optional[bool] = Field(default=None, description="Whether reviewer recommends it")