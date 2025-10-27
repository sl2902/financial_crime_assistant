"""LLM-based entity extractor for SEC financial crime documents"""

import os
import json
from typing import Any, List, Dict, Optional
from loguru import logger
import glob
import aiofiles
import asyncio

from src.config.model_config import (
    EntityExtractorConfig, 
    MODEL_PRESETS, 
    get_preset_config,
    processed_file_path,
    save_path
)
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

        **CRITICAL INSTRUCTIONS:**

        **PERSONS - Extract ONLY:**
        - Defendants (individuals charged with violations)
        - Respondents (individuals named in the complaint)
        - Officers/employees of companies being charged (CEO, CFO, etc.)

        **DO NOT EXTRACT:**
        - SEC staff (attorneys, investigators, supervisors)
        - Judges or court officials
        - Auditors or external parties (unless they are defendants)

        **COMPANIES - Extract:**
        - All companies charged or named as respondents
        - Companies affiliated with defendants

        **CRIME TYPES - Identify from this list:**
        - Ponzi Scheme
        - Insider Trading
        - Securities Fraud
        - Wire Fraud
        - Money Laundering
        - Embezzlement
        - Market Manipulation
        - Bribery/FCPA
        - Fraud (General)
        - Financial Crime (Other)

        **PENALTIES - Extract:**
        - Type (Monetary, Disgorgement, Injunction, Bar, etc.)
        - Amount (parse carefully: "$1.5 million" = 1500000.0)
        - Description
        - Recipient (person or company name)

        **COMPANY AFFILIATIONS:**
        - For each person, list the companies they are associated with
        - Example: If "John Smith, CEO of Acme Corp" → company_affiliations: ["Acme Corp"]

        **DATE FORMAT:**
        - Use YYYY-MM-DD format for filing_date
        - Extract from the document or use metadata date

        Extract all entities now using the tool.
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
            elif config.provider == "openai" or config.provider == "openai_async":
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
        
        if self.config.provider == "openai_async":
            from openai import AsyncOpenAI
            return AsyncOpenAI(api_key=api_key)
        
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
        try:
            if self.config.provider == "anthropic":
                extracted_data = self._call_anthropic_with_tools(text, lr_number, metadata)
            elif self.config.provider == "openai" or self.config.provider == "openai_async":
                extracted_data = self._call_openai_with_tools(text, lr_number, metadata)
            elif self.config.provider == "google":
                # Google uses function calling (similar to tools)
                extracted_data = self._call_google_with_tools(text, lr_number, metadata)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")

            # Build ExtractedEntities with validation
            return self._build_extracted_entities(extracted_data, text, lr_number, metadata)

        except Exception as e:
            logger.info(f"Error extracting entities from {lr_number}: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                return ExtractedEntities(
                    lr_number=lr_number,
                    case=Case(
                        lr_number=lr_number,
                        title=f"Error processing {lr_number}",
                        crime_types=[CrimeType.FINANCIAL_CRIME_OTHER],
                        filing_date=metadata.get("filing_date", "2024-01-01"),
                        url=metadata.get("url", "https://www.sec.gov"),
                    ),
                    raw_text=text[:500],
                )
            except:
                return None
    
    async def extract_entities_async(
            self, text: str, lr_number: str, metadata: Optional[dict] = None
    ) -> ExtractedEntities:
        """Async version of extract_entities."""
        try:
            # Call appropriate provider asynchronously
            if self.config.provider == "anthropic":
                extracted_data = await self._call_anthropic_with_tools_async(text, lr_number, metadata)
            elif self.config.provider == "openai" or self.config.provider == "openai_async":
                extracted_data = await self._call_openai_with_tools_async(text, lr_number, metadata)
            elif self.config.provider == "google":
                extracted_data = await self._call_google_with_tools_async(text, lr_number, metadata)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")

            return self._build_extracted_entities(extracted_data, text, lr_number, metadata)

        except Exception as e:
            logger.error(f"Error extracting entities from {lr_number}: {e}")
            # Return minimal fallback
            try:
                return ExtractedEntities(
                    lr_number=lr_number,
                    case=Case(
                        lr_number=lr_number,
                        title=f"Error processing {lr_number}",
                        crime_types=[CrimeType.FINANCIAL_CRIME_OTHER],
                        filing_date=metadata.get("filing_date", "2024-01-01"),
                        url=metadata.get("url", "https://www.sec.gov"),
                    ),
                    raw_text=text[:500],
                )
            except:
                return None
    
    def _get_tool_schema(self) -> Dict:
        """Generate tool schema from Pydantic models"""

        return {
            "name": "extract_entities",
            "description": "ONLY extract defendants, respondents, or individuals charged with violations. DO NOT extract SEC staff, investigators, attorneys, or supervisors.",
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
                                "role": {"type": "string", "description": "Include person's role if mentioned (CEO, CFO, Founder, etc.)"},
                                "company_affiliations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List Companies associated/affiliated with this person"
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
    
    async def _call_openai_with_tools_async(self, text: str, lr_number: str, metadata: Optional[Dict]) -> Dict:
        """Async OpenAI API call."""
        tool_schema = self._get_tool_schema()
        
        function_def = {
            "type": "function",
            "function": {
                "name": tool_schema["name"],
                "description": tool_schema["description"],
                "parameters": tool_schema["input_schema"]
            }
        }
        
        prompt = PROMPT.format(lr_number=lr_number, metadata=metadata, text=text)
        
        # Use async OpenAI client
        response = await self.client.chat.completions.create(
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
            self, documents: List[Dict], save_path: Optional[str] = None
    ) -> List[ExtractedEntities]:
        """Extract entities from multiple documents.

        Args:
            documents: List of dicts with 'text', 'lr_number', 'metadata' keys
            save_path: Optional path to save extracted entities as JSONL

        Returns:
            List of ExtractedEntities objects
        """
        results = []

        SEC_KEYWORDS = ['attorney', 'investigator', 'supervisor', 'trial counsel', 
                'regional office', 'litigation', 'sec ']


        for i, doc in enumerate(documents):
            try:
                logger.info(f"Processing {i+1}/{len(documents)}: {doc['lr_no']}")
                if ',' in doc['lr_no']:
                    logger.warning(f"Document{doc['lr_no']} has invalid LR number. Attempt to clean it or reject the document")
                    doc["lr_no"] = doc["lr_no"].split(", ")[0].strip()
                    logger.info("Successfully cleaned the document")

                metadata = {
                        'lr_no': doc.get('lr_no', ''),
                        'date': doc.get('date', ''),
                        'url': doc.get('url', None) or doc.get('main_link', ''),
                        'content_length': doc.get('content_length'),
                        'crime_type': doc.get('crime_type', []),
                        'amounts': doc.get('amounts', []),
                        'penalty_category': doc.get('penalty_category', 'Uknown'),
                        'people_mentioned': doc.get('people_mentioned', []),
                        'title': doc.get('title', ''),
                        'see_also': doc.get('see_also', ''),
                        'source': 'SEC'
                }

                entities = self.extract_entities(
                    text=doc["content"],
                    lr_number=doc["lr_no"],
                    metadata=metadata,
                )
                entities.persons = [
                    p for p in entities.persons
                    if not any(keyword in (p.role or "").lower() for keyword in SEC_KEYWORDS)
                ]

                results.append(entities)
                stats = entities.summary_stats()
                logger.info(f"  Extracted: {stats['num_persons']} persons, "
                    f"{stats['num_companies']} companies, "
                    f"{stats['num_penalties']} penalties")
            except Exception as e:
                logger.error(f" Failed to process {doc['lr_no']}: {e}")
                continue
        
        if save_path:
            with open(save_path, "w") as f:
                for entity in results:
                    f.write(entity.model_dump_json() + "\n")
            logger.info(f"\nSaved {len(results)} extracted entities to {save_path}")

        return results
    
    async def batch_extract_async(
        self, 
        documents: List[Dict], 
        save_path: Optional[str] = None,
        concurrency: int = 10
    ) -> List[ExtractedEntities]:
        """Async batch extraction with concurrency control."""
        
        SEC_KEYWORDS = ['attorney', 'investigator', 'supervisor', 'trial counsel', 
                        'regional office', 'litigation', 'sec ']
        
        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_doc(doc, index):
            async with semaphore:
                logger.info(f"Processing {index+1}/{len(documents)}: {doc['lr_no']}")

                if ',' in doc['lr_no']:
                    logger.warning(f"Document{doc['lr_no']} has invalid LR number. Attempt to clean it or reject the document")
                    doc["lr_no"] = doc["lr_no"].split(", ")[0].strip()
                    logger.info("Successfully cleaned the document")
                
                metadata = {
                    'lr_no': doc.get('lr_no', ''),
                    'date': doc.get('date', ''),
                    'url': doc.get('url', None) or doc.get('main_link', ''),
                    'content_length': doc.get('content_length'),
                    'crime_type': doc.get('crime_type', []),
                    'amounts': doc.get('amounts', []),
                    'penalty_category': doc.get('penalty_category', 'Uknown'),
                    'people_mentioned': doc.get('people_mentioned', []),
                    'title': doc.get('title', ''),
                    'see_also': doc.get('see_also', ''),
                    'source': 'SEC'
                }
                
                try:
                    entities = await self.extract_entities_async(
                        text=doc["content"],
                        lr_number=doc["lr_no"],
                        metadata=metadata,
                    )
                    
                    if entities:
                        # Filter SEC staff
                        entities.persons = [
                            p for p in entities.persons 
                            if not any(keyword in (p.role or "").lower() 
                                    for keyword in SEC_KEYWORDS)
                        ]
                        
                        stats = entities.summary_stats()
                        logger.info(f"  {doc['lr_no']}: {stats['num_persons']} persons, "
                                f"{stats['num_companies']} companies, "
                                f"{stats['num_penalties']} penalties")
                        return entities
                    
                except Exception as e:
                    logger.error(f"  Failed {doc['lr_no']}: {e}")
                    return None
                    # async with aiofiles.open("failed_extractions.txt", "a") as f:
                    #     await f.write(f"{doc['lr_no']}\n")
                    # return None
        
        # Process all documents concurrently
        tasks = [process_doc(doc, i) for i, doc in enumerate(documents)]
        results = await asyncio.gather(*tasks)

        # for result in results:
        #     stats = result.summary_stats()
        #     logger.info(
        #         f"Extracted: {stats['num_persons']} persons, "
        #         f"{stats['num_companies']} companies, "
        #         f"{stats['num_penalties']} penalties"
        #     )
        
        # Filter out None values (failed extractions)
        results = [r for r in results if r is not None]
        
        if save_path:
            async with aiofiles.open(save_path, "w") as f:
                for entity in results:
                    await f.write(entity.model_dump_json() + "\n")
            logger.info(f"\n Saved {len(results)} entities to {save_path}")
        
        return results
    
async def main():
    config = EntityExtractorConfig(
        provider="openai_async",
        model="gpt-4o-mini"
    )
    extractor = EntityExtractor(config=config)

    batch_files = sorted(glob.glob(f'{processed_file_path}/sec_releases_batch*_clean.json'))
    for i, batch_file in enumerate(batch_files, start=1):
        logger.info(f"Processing batch file {batch_file}")
        file = f'sec_releases_batch_{i}_kg.jsonl'
        with open(batch_file, 'r') as f:
            data = json.load(f)
            results = await extractor.batch_extract_async(
                documents=data["releases"],
                save_path=f"{save_path}/{file}",
                concurrency=10
            )
        
        logger.info(f"Processed {len(results)} documents")

    

if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info(f"Example 1: Using budget preset ({MODEL_PRESETS.get('budget').get('openai_async')})")
    logger.info("=" * 60)
    
    config = get_preset_config("budget", "openai")
    extractor = EntityExtractor(config=config)
    
    logger.info(f"Provider: {extractor.config.provider}")
    logger.info(f"Model: {extractor.config.model}")

    # logger.info("\n" + "=" * 60)
    # logger.info("Example 2: Direct initialization")
    # logger.info("=" * 60)
    
    # extractor = EntityExtractor(
    #     provider="anthropic",
    #     model="claude-sonnet-4-20250514"
    # )

    # sample_text = """
    # SEC Charges Former CEO with Securities Fraud
    
    # The Securities and Exchange Commission today announced charges against John Smith,
    # former CEO of TechCorp Inc. (NASDAQ: TECH), for securities fraud. Smith allegedly
    # inflated revenue by $50 million over three years. TechCorp agreed to pay a
    # $5 million civil penalty and implement enhanced controls. Smith was also barred
    # from serving as an officer or director of a public company.
    # """

    # logger.info("\nExtracting entities...")
    # entities = extractor.extract_entities(
    #     text=sample_text,
    #     lr_number="LR-25000",
    #     metadata={
    #         "url": "https://www.sec.gov/litigation/litreleases/lr25000.htm",
    #         "filing_date": "2024-10-24"
    #     },
    # )

    # logger.info("\n" + "=" * 60)
    # logger.info("EXTRACTION RESULTS")
    # logger.info("=" * 60)
    # logger.info(json.dumps(entities.summary_stats(), indent=2))
    # persons = [p.name for p in entities.persons]
    # companies = [c.name for c in entities.companies]
    # penalties = [f"${p.amount:,.0f}" if p.amount else p.penalty_type.value 
    #                      for p in entities.penalties]
    # logger.info(f"\nPersons: {persons}")
    # logger.info(f"\nCompanies: {companies}")
    # logger.info(f"\nPenalties: {penalties}")


    # batch_files = sorted(glob.glob(f'{processed_file_path}/sec_releases_batch*_clean.json'))
    # for i, batch_file in enumerate(batch_files[1:], start=1):
    #     logger.info(f"Processing batch file {batch_file}")
    #     file = f'sec_releases_batch_{i}_kg.jsonl'
    #     with open(batch_file, 'r') as f:
    #         data = json.load(f)
    #         extractor.batch_extract(data["releases"], f"{save_path}/{file}")
    asyncio.run(main())



