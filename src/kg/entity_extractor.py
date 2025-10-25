"""LLM-based entity extractor for SEC financial crime documents"""

import os
import json
from typing import Any, List, Dict, Optional
from loguru import logger

from src.config.model_config import EntityExtractorConfig
from src.schemas.entity_schemas import (
    Case,
    Company,
    CrimeType,
    ExtractedEntities,
    Penalty,
    PenaltyType,
    Person,
)

PROMPT = """Analyze this SEC enforcement document and extract all relevant entities.

        Document: {lr_number}
        URL: metadata.get('url') if {metadata} else 'N/A'

        Text:
        {text}

        Instructions:
        - Extract ALL persons charged or involved
        - Extract ALL companies mentioned
        - Identify all crime types
        - Parse monetary amounts carefully
        - Extract all penalties with recipients
        - Use YYYY-MM-DD date format
        """

class EntityExtractor:
    """Extract structured entities from SEC financial crime documents using LLMs
    
    Supports multiple LLM providers: Anthropic (Claude), OpenAI (GPT), Google (Gemini)
    """

    def __init__(
            self,
            config: Optional[EntityExtractorConfig] = None,
            provider: Optional[str] = None,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
    ):
        """Initialize the entity extractor.

        Args:
            config: EntityExtractorConfig object (recommended)
            provider: LLM provider ("anthropic", "openai", "google") - overrides config
            model: Model name - overrides config
            api_key: API key - overrides config/env
            
        Examples:
            # Using config (recommended)
            config = EntityExtractorConfig(provider="anthropic", model="claude-haiku-3.5")
            extractor = EntityExtractor(config=config)
            
            # Using presets
            extractor = EntityExtractor(config=get_preset_config("budget", "openai"))
            
            # Direct parameters
            extractor = EntityExtractor(provider="anthropic", model="claude-sonnet-4")
        """
        if config is None:
            provider = EntityExtractorConfig(
                provider or "anthropic",
                model=model or "claude-sonnet-4-2025051",
            )
        
        if provider:
            config.provider = provider
        
        if model:
            config.model = model
        
        if api_key:
            if config.provider == provider:
                config.anthropic_api_key = api_key
            elif config.provider == "openai":
                config.openai_api_key = api_key
            elif config.provider == "google":
                config.google_api_key = api_key
        
        self.config = config
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client based on provider"""
        api_key = self.config.get_api_key()

        if self.config.provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic(api_key=api_key)
        
        if self.config.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=api_key)
        
        if self.config.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.config.model)
        
        raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def extract_entities(
            self, text: str, lr_number: str, metadata: Optional[Dict] = None
    ) -> ExtractedEntities:
        """Extract entities from SEC document text

        Args:
            text: Raw document text
            lr_number: Litigation Release number
            metadata: Optional metadata (filing_date, url, etc.)

        Returns:
            ExtractedEntities object with all extracted entities
        """
        prompt = self._build_extraction_prompt(text, lr_number, metadata)

        try:
            if self.config.provider == "anthropic":
                response_text = self._call_anthropic(prompt)
            elif self.config.provider == "openai":
                response_text = self._call_openai(prompt)
            elif self.config.provider == "google":
                response_text = self._call_google(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
            
            json_text = self._extract_json(response_text)
            extracted_data = json.loads(json_text)
    
            return self._build_extracted_entities(extracted_data, text, lr_number, metadata)
        
        except Exception as e:
            logger.info(f"Error extracting entities from {lr_number}: {e}")
            # return minimal valid structure
            return ExtractedEntities(
                lr_number=lr_number,
                case=Case(
                    lr_number=lr_number,
                    title=f"Error processing {lr_number}",
                    crime_types=[CrimeType.FINANCIAL_CRIME_OTHER],
                    filing_date=metadata.get("filing_date", "1900-01-01"),
                    url=metadata.get("url", "https://www.sec.gov"),
                ),
                raw_text=text[:500],  # Store snippet for debugging
            )
    
    def _get_tool_schema(self) -> Dict:
        """Generate tool schema from Pydantic models"""

        return {
            "name": "extract_entities",
            "description": "Extract structured entities (persons, companies, case details, penalties) from SEC enforcement document",
            "input_schema": {
                "type": "object",
                "properties": {
                    "persons": {
                        "type": "array",
                        "description": "Individuals charged or involved in the case",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Full name"},
                                "role": {"type": "string", "description": "Title/role (e.g., CEO, CFO)"},
                                "company_affiliations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Companies associated with this person"
                                }
                            },
                            "required": ["name"]
                        }
                    },
                    "companies": {
                        "type": "array",
                        "description": "Companies involved in the case",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Company name"},
                                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                                "industry": {"type": "string", "description": "Industry/sector"},
                                "description": {"type": "string", "description": "Brief description"}
                            },
                            "required": ["name"]
                        }
                    },
                    "case": {
                        "type": "object",
                        "description": "Case details",
                        "properties": {
                            "title": {"type": "string", "description": "Case title"},
                            "crime_types": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "Ponzi Scheme",
                                        "Insider Trading",
                                        "Securities Fraud",
                                        "Wire Fraud",
                                        "Money Laundering",
                                        "Embezzlement",
                                        "Market Manipulation",
                                        "Bribery/FCPA",
                                        "Fraud (General)",
                                        "Financial Crime (Other)"
                                    ]
                                },
                                "description": "Types of financial crimes"
                            },
                            "filing_date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                            "summary": {"type": "string", "description": "2-3 sentence case summary"}
                        },
                        "required": ["title", "crime_types", "filing_date"]
                    },
                    "penalties": {
                        "type": "array",
                        "description": "Penalties imposed",
                        "items": {
                            "type": "object",
                            "properties": {
                                "penalty_type": {
                                    "type": "string",
                                    "enum": ["Monetary", "Injunction", "Suspension", "Bar", "Disgorgement", "Cease and Desist", "Other"],
                                    "description": "Type of penalty"
                                },
                                "amount": {"type": "number", "description": "Monetary amount if applicable"},
                                "description": {"type": "string", "description": "Penalty description"},
                                "recipient": {"type": "string", "description": "Person or company receiving penalty"}
                            },
                            "required": ["penalty_type", "description"]
                        }
                    }
                },
                "required": ["persons", "companies", "case", "penalties"]
            }
        }
    
    def _call_anthropic_with_tools(self, text: str, lr_number: str, metadata: Optional[Dict]) -> Dict:
        """Call Anthropic API with tool calling"""

        tool_schema = self._get_tool_schema()

        prompt = PROMPT.format(lr_number=lr_number, metadata=metadata, text=text)
        
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            tools=[tool_schema],
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )

        for content in response.content:
            if content.type == 'tool_use':
                return content.input
        
        raise ValueError("No tool use found in response")
    
    def _call_openai_with_tools(self, text: str, lr_number: str, metadata: Optional[Dict]) -> Dict:
        """Call OpenAI API with function calling."""

        tool_schema = self._get_tool_schema()

        prompt = PROMPT.format(lr_number=lr_number, metadata=metadata, text=text)

        function_def = {
            "type": "function",
            "function": {
                "name": tool_schema["name"],
                "description": tool_schema["description"],
                "parameters": tool_schema["input_schema"],
            }
        }

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            tools=[function_def],
            tool_choice={
                "type": "function",
                "function": {
                    "name": "extract_entities",
                }
            },
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)

    def _call_google_with_tools(self, text: str, lr_number: str, metadata: Optional[dict]) -> dict:
        """Call Google Gemini API with function calling."""
        # For Gemini, we'll use JSON mode as fallback since function calling setup is different
        # This is a simplified version - full implementation would require google.generativeai schema
        prompt = f"""Analyze this SEC enforcement document and extract entities in JSON format.

                Document: {lr_number}

                Text:
                {text}

                Return a JSON object with: persons, companies, case, penalties
                Use the exact schema structure with all required fields.
                """
        
        response = self.client.generate_content(
            prompt,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            }
        )

        json_text = self._extract_json(response.text)
        return json.loads(json_text)
    
    def _build_extracted_entities(
            self, data: dict, text: str, lr_number: str, metadata: Optional[dict] = None
    ) -> ExtractedEntities:
        """Build and validate ExtractedEntities from parsed JSON.

        Args:
            data: Parsed JSON data from LLM
            text: Original document text
            lr_number: LR number
            metadata: Optional metadata

        Returns:
            Validated ExtractedEntities object
        """

        persons = [
            Person(
                name=p["name"],
                role=p.get("role"),
                company_affiliations=p.get("company_affiliations", []),
            )
            for p in data.get("persons", [])
        ]

        companies = [
            Company(
                name=c["name"],
                ticker=c.get("ticker"),
                industry=c.get("industry"),
                description=c.get("description"),
            )
            for c in data.get("companies", [])
        ]

        case_data = data.get("case", {})

        crime_types = []
        for ct in case_data.get("crime_types", []):
            try:
                crime_types.append(CrimeType(ct))
            except ValueError:
                crime_types.append(CrimeType.FINANCIAL_CRIME_OTHER)
        
        if not crime_types:
            crime_types = [CrimeType.FINANCIAL_CRIME_OTHER]
        
        case = Case(
            lr_number=lr_number,
            title=case_data.get("title", f"Case {lr_number}"),
            crime_types=crime_types,
            filing_date=case_data.get("filing_date"),
            summary=case_data.get("summary"),
            url=case_data.get("url") or (metadata.get("url") if metadata else None),
        )

        penalties = []
        for p in data.get("penalties", []):
            try:
                penalty_type = PenaltyType(p["penalty_type"])
            except (ValueError, KeyError):
                penalty_type = PenaltyType.OTHER
            
            penalties.append(
                Penalty(
                    penalty_type=penalty_type,
                    amount=p.get("amount"),
                    description=p.get("description", ""),
                    recipient=p.get("recipient"),
                )
            )
        
        return ExtractedEntities(
            lr_number=lr_number,
            persons=persons,
            companies=companies,
            case=case,
            penalties=penalties,
            raw_text=text[:1000],
        )
    
    def batch_extract(
            self, documents: list[dict], save_path: Optional[str] = None
    ) -> List[ExtractedEntities]:
        """Extract entities from multiple documents.

        Args:
            documents: List of dicts with 'text', 'lr_number', 'metadata' keys
            save_path: Optional path to save extracted entities as JSONL

        Returns:
            List of ExtractedEntities objects
        """
        results = []

        for i, doc in enumerate(documents):
            logger.info(f"Processing {i+1}/{len(documents)}: {doc['lr_number']}")

            entities = self.extract_entities(
                text=doc["text"],
                lr_number=doc["lr_number"],
                metadata=doc.get("metadata"),
            )

            results.append(entities)

            stats = entities.summary_stats()
            logger.info(f"  Extracted: {stats['num_persons']} persons, "
                  f"{stats['num_companies']} companies, "
                  f"{stats['num_penalties']} penalties")
        
        if save_path:
            with open(save_path, "w") as f:
                for entity in results:
                    f.write(entity.model_dump_json() + "\n")
            logger.info(f"\nSaved {len(results)} extracted entities to {save_path}")

        return results
    

if __name__ == "__main__":
    




