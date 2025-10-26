"""Test entity extraction on sample documents."""

import json
import sys
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kg.entity_extractor import EntityExtractor
from src.config.model_config import EntityExtractorConfig, MODEL_PRESETS, get_preset_config


def test_extraction_sample():
    """Test entity extraction on a sample SEC document."""
    logger.info(" Testing Entity Extraction\n")

    # Sample SEC document text
    sample_doc = {
        "lr_number": "LR-25123",
        "text": """
Securities and Exchange Commission v. Michael Thompson and GlobalTech Solutions Inc.

The Securities and Exchange Commission announced fraud charges against Michael Thompson, 
former Chief Financial Officer of GlobalTech Solutions Inc. (NYSE: GLBT), for orchestrating 
a $45 million accounting fraud scheme.

From 2019 to 2022, Thompson allegedly manipulated the company's financial statements by 
inflating revenue through fictitious sales transactions and concealing material liabilities. 
The scheme allowed GlobalTech to meet analyst expectations and maintain its stock price.

Thompson, 52, of San Francisco, California, also allegedly engaged in insider trading by 
selling $2.3 million worth of company stock while aware of the fraudulent accounting practices.

The SEC's complaint, filed in federal court in Northern California, charges Thompson with 
violations of the antifraud, reporting, and internal controls provisions of the federal 
securities laws. GlobalTech Solutions consented to pay a $10 million civil penalty without 
admitting or denying the allegations.

Additionally, the SEC is seeking:
- Permanent injunction against Thompson
- Disgorgement of ill-gotten gains of $2.3 million plus prejudgment interest
- Civil penalties
- Officer and director bar

GlobalTech Solutions, headquartered in Palo Alto, California, develops enterprise software 
solutions for the healthcare industry. The company has agreed to implement enhanced internal 
controls and hire an independent compliance monitor for three years.

The SEC's investigation was conducted by Sarah Chen and David Rodriguez of the San Francisco 
Regional Office.
        """,
        "metadata": {
            "url": "https://www.sec.gov/litigation/litreleases/2023/lr25123.htm",
            "filing_date": "2023-08-15",
        },
    }

    # Initialize extractor
    config = get_preset_config("budget", "openai")
    extractor = EntityExtractor(config=config)

    # Extract entities
    logger.info("📄 Processing sample document...")
    entities = extractor.extract_entities(
        text=sample_doc["text"],
        lr_number=sample_doc["lr_number"],
        metadata=sample_doc["metadata"],
    )

    # Display results
    logger.info("\n" + "=" * 70)
    logger.info("EXTRACTION RESULTS")
    logger.info("=" * 70)

    logger.info(f"\n📋 Case: {entities.case.lr_number}")
    logger.info(f"   Title: {entities.case.title}")
    logger.info(f"   Crime Types: {[ct.value for ct in entities.case.crime_types]}")
    logger.info(f"   Filing Date: {entities.case.filing_date}")
    logger.info(f"   Summary: {entities.case.summary[:200] if entities.case.summary else 'N/A'}...")

    logger.info(f"\n👥 Persons Extracted ({len(entities.persons)}):")
    for person in entities.persons:
        logger.info(f"   • {person.name}")
        logger.info(f"     Role: {person.role or 'N/A'}")
        logger.info(f"     Companies: {', '.join(person.company_affiliations) or 'N/A'}")

    logger.info(f"\n🏢 Companies Extracted ({len(entities.companies)}):")
    for company in entities.companies:
        logger.info(f"   • {company.name}")
        logger.info(f"     Ticker: {company.ticker or 'N/A'}")
        logger.info(f"     Industry: {company.industry or 'N/A'}")

    logger.info(f"\n💰 Penalties Extracted ({len(entities.penalties)}):")
    for penalty in entities.penalties:
        logger.info(f"   • {penalty.penalty_type.value}")
        if penalty.amount:
            logger.info(f"     Amount: ${penalty.amount:,.0f}")
        logger.info(f"     Recipient: {penalty.recipient or 'N/A'}")
        logger.info(f"     Description: {penalty.description[:100]}...")

    logger.info("\n" + "=" * 70)

    # Summary stats
    stats = entities.summary_stats()
    logger.info("\n Summary Statistics:")
    logger.info(json.dumps(stats, indent=2))

    # Validate extraction quality
    logger.info("\n Validation:")
    assert len(entities.persons) > 0, "No persons extracted!"
    assert len(entities.companies) > 0, "No companies extracted!"
    assert len(entities.penalties) > 0, "No penalties extracted!"
    assert entities.case.title, "No case title!"
    logger.info("   All validations passed!")

    return entities


def test_batch_extraction():
    """Test batch extraction on multiple documents."""
    logger.info("\n\n Testing Batch Extraction\n")

    # Multiple sample documents
    documents = [
        {
            "lr_number": "LR-25001",
            "text": """SEC Charges Investment Advisor with Ponzi Scheme
            
            Robert Wilson operated a $30 million Ponzi scheme through his firm, 
            Capital Advisors LLC. Wilson promised returns of 15% annually but used 
            new investor funds to pay earlier investors. Wilson agreed to pay 
            $5 million in disgorgement and was barred from the securities industry.""",
            "metadata": {"url": "https://www.sec.gov/litigation/litreleases/lr25001.htm"},
        },
        {
            "lr_number": "LR-25002",
            "text": """Insider Trading Charges Against Tech Executive
            
            The SEC charged Jennifer Martinez, VP of Engineering at DataFlow Inc. 
            (NASDAQ: DATA), with insider trading. Martinez sold $500,000 of company 
            stock before the announcement of poor earnings. She agreed to settle for 
            $150,000 in penalties and disgorgement.""",
            "metadata": {"url": "https://www.sec.gov/litigation/litreleases/lr25002.htm"},
        },
    ]

    config = get_preset_config("budget", "openai")
    extractor = EntityExtractor(config=config)
    results = extractor.batch_extract(
        documents=documents, save_path="data/kg/tests/test_extraction.jsonl"
    )

    logger.info(f"\n Extracted entities from {len(results)} documents")
    for result in results:
        logger.info(f"   • {result.lr_number}: {result.summary_stats()}")


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("ENTITY EXTRACTION TEST SUITE")
    logger.info("=" * 70 + "\n")

    # Test 1: Single document extraction
    try:
        test_extraction_sample()
    except Exception as e:
        logger.error(f"\n Test 1 failed: {e}")
        import traceback

        traceback.print_exc()

    # Test 2: Batch extraction
    try:
        test_batch_extraction()
    except Exception as e:
        logger.info(f"\n Test 2 failed: {e}")
        import traceback

        traceback.print_exc()

    logger.info("\n" + "=" * 70)
    logger.info(" ALL TESTS COMPLETED")
    logger.info("=" * 70)