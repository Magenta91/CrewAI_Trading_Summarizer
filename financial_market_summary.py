#!/usr/bin/env python3
"""
CrewAI Financial Market Summary Script
=====================================

A comprehensive multi-agent system for generating daily US financial market summaries
after market close (01:30 IST) using CrewAI framework.

Setup Instructions:
1. Ensure Python 3.10+ is installed
2. Install dependencies: pip install -r requirements.txt
3. Configure .env file with required API keys (see .env.example)
4. Run: python financial_market_summary.py

Author: CrewAI Financial Bot
Version: 2.0
"""

import os
import sys
import json
import asyncio
import logging
import schedule
import time
import pytz
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# CrewAI imports
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import TavilySearchTool, SerperDevTool
from crewai.tools import BaseTool

# LiteLLM for model management
import litellm

# Telegram imports
from telegram import Bot
from telegram.error import TelegramError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_summary.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ConfigManager:
    """Manages configuration and environment variables"""
    
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_channel_id = os.getenv("TELEGRAM_CHANNEL_ID")
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate required environment variables"""
        required_vars = {
            "GOOGLE_API_KEY": self.google_api_key,
            "TELEGRAM_BOT_TOKEN": self.telegram_bot_token,
            "TELEGRAM_CHANNEL_ID": self.telegram_channel_id
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Check for at least one search API
        if not (self.tavily_api_key or self.serper_api_key):
            raise ValueError("Either TAVILY_API_KEY or SERPER_API_KEY must be provided")
        
        logger.info("Configuration validated successfully")

class LLMManager:
    """Manages LLM configuration and interactions"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.model = "gemini/gemini-1.5-flash"  # Updated to Gemini model (confirmed working)
        
        # Configure LiteLLM globally for Gemini
        os.environ["GOOGLE_API_KEY"] = config.google_api_key
        os.environ["GEMINI_API_KEY"] = config.google_api_key  # Alternative env var
        litellm.set_verbose = False
        
    def get_llm_config(self):
        """Returns LLM configuration for CrewAI agents"""
        # For newer CrewAI versions, use string model name
        return self.model

class SearchToolManager:
    """Manages search tools (Tavily/Serper)"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.search_tool = self._initialize_search_tool()
    
    def _initialize_search_tool(self):
        """Initialize the appropriate search tool"""
        if self.config.tavily_api_key:
            logger.info("Using Tavily search tool")
            return TavilySearchTool(api_key=self.config.tavily_api_key)
        elif self.config.serper_api_key:
            logger.info("Using Serper search tool")
            return SerperDevTool(api_key=self.config.serper_api_key)
        else:
            raise ValueError("No search tool API key available")



class TelegramManager:
    """Manages Telegram bot interactions"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.bot = Bot(token=config.telegram_bot_token)
        self.channel_id = config.telegram_channel_id
    
    async def send_message(self, message: str, parse_mode: str = "Markdown") -> Dict[str, Any]:
        """Send a message to the Telegram channel"""
        try:
            await self.bot.send_message(
                chat_id=self.channel_id,
                text=message,
                parse_mode=parse_mode
            )
            logger.info("Message sent successfully to Telegram")
            return {"status": "success", "message": "Message sent successfully"}
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {str(e)}")
            return {"status": "error", "message": f"Failed to send message: {str(e)}"}
    
    async def send_photo(self, photo_url: str, caption: str = "") -> Dict[str, Any]:
        """Send a photo to the Telegram channel"""
        try:
            await self.bot.send_photo(
                chat_id=self.channel_id,
                photo=photo_url,
                caption=caption,
                parse_mode="Markdown"
            )
            logger.info(f"Photo sent successfully: {photo_url}")
            return {"status": "success", "message": "Photo sent successfully"}
        except TelegramError as e:
            logger.error(f"Failed to send photo: {str(e)}")
            return {"status": "error", "message": f"Failed to send photo: {str(e)}"}

class FinancialSummaryCrewAI:
    """Main CrewAI system for financial market summaries"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.llm_manager = LLMManager(self.config)
        self.search_manager = SearchToolManager(self.config)
        self.telegram_manager = TelegramManager(self.config)

        
        # Initialize agents and tasks
        self.agents = self._create_agents()
        self.tasks = self._create_tasks()
        self.crew = self._create_crew()
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create all CrewAI agents"""
        llm_config = self.llm_manager.get_llm_config()
        
        agents = {
            "search_agent": Agent(
                role="Financial News Researcher",
                goal="Search for the latest US financial market news and data from the past hour",
                backstory="""You are an expert financial researcher with deep knowledge of US markets.
                You specialize in finding the most relevant and timely financial news, market movements,
                and trading activity. You know how to identify credible sources and filter out noise.""",
                tools=[self.search_manager.search_tool],
                llm=llm_config,
                verbose=True,
                allow_delegation=False,
                max_iter=3
            ),
            
            "summary_agent": Agent(
                role="Financial News Summarizer",
                goal="Create concise, informative summaries of financial news under 500 words",
                backstory="""You are a skilled financial journalist who excels at distilling complex
                market information into clear, actionable summaries. You understand what matters most
                to investors and traders, focusing on market-moving events and key trends.""",
                llm=llm_config,
                verbose=True,
                allow_delegation=False,
                max_iter=2
            ),
            
            "formatting_agent": Agent(
                role="Content Formatter and Visual Enhancer",
                goal="Format summaries with relevant charts and images for maximum impact",
                backstory="""You are a content specialist who knows how to enhance financial content
                with appropriate visual elements. You understand the importance of charts, graphs,
                and images in conveying market information effectively.""",
                tools=[self.search_manager.search_tool],
                llm=llm_config,
                verbose=True,
                allow_delegation=False,
                max_iter=2
            ),
            
            "translation_agent": Agent(
                role="Multilingual Financial Translator",
                goal="Translate financial content accurately into Arabic, Hindi, and Hebrew",
                backstory="""You are an expert translator specializing in financial terminology
                across multiple languages. You maintain accuracy while preserving the original
                formatting and ensuring cultural appropriateness of financial content.""",
                llm=llm_config,
                verbose=True,
                allow_delegation=False,
                max_iter=2
            ),
            
            "distribution_agent": Agent(
                role="Content Distribution Specialist",
                goal="Distribute financial summaries through Telegram channels effectively",
                backstory="""You are responsible for the final delivery of financial content
                to end users. You ensure proper formatting for messaging platforms and
                handle any distribution issues that may arise.""",
                llm=llm_config,
                verbose=True,
                allow_delegation=False,
                max_iter=2
            )
        }
        
        logger.info("All agents created successfully")
        return agents
    
    def _create_tasks(self) -> List[Task]:
        """Create all CrewAI tasks with guardrails"""
        
        search_task = Task(
            description="""Search for US financial market news from the last 2 hours focusing on:
            - Major stock market movements (S&P 500, NASDAQ, Dow Jones)
            - Significant earnings reports or corporate announcements
            - Federal Reserve or economic policy news
            - Currency and commodity market updates
            - Any market-moving events or breaking financial news
            
            Provide a comprehensive list of the most relevant findings with sources.""",
            expected_output="""A detailed report containing:
            - List of relevant news items with titles, sources, and summaries
            - Key market movements and statistics
            - Search timestamp
            - Number of sources found""",
            agent=self.agents["search_agent"]
        )
        
        summary_task = Task(
            description="""Create a comprehensive yet concise summary (400-500 words) of the financial news.
            Structure the summary with:
            1. Market Overview (major indices performance)
            2. Key News Highlights (2-3 most important stories)
            3. Sector Performance (notable winners/losers)
            4. Looking Ahead (what to watch tomorrow)
            
            Use clear, professional language suitable for both novice and experienced investors.""",
            expected_output="A well-structured English summary between 400-500 words",
            agent=self.agents["summary_agent"],
            context=[search_task]
        )
        
        formatting_task = Task(
            description="""Format the summary for optimal readability and find 2 relevant financial charts or images.
            
            Tasks:
            1. Format the summary with proper Markdown headers and bullet points
            2. Search for and identify 2 relevant charts/images (market charts, company logos, etc.)
            3. Embed the images using Markdown syntax at logical positions
            4. Ensure the final format is suitable for Telegram messaging
            
            Focus on charts showing market performance, sector analysis, or key company data.
            
            Provide the output in this format:
            FORMATTED_SUMMARY: [Markdown-formatted summary with embedded images]
            IMAGE_URLS: [url1, url2]
            IMAGE_DESCRIPTIONS: [description1, description2]""",
            expected_output="""A formatted report containing:
            - Markdown-formatted summary with embedded images
            - List of 2 relevant image URLs
            - Descriptions for each image""",
            agent=self.agents["formatting_agent"],
            context=[summary_task]
        )
        
        translation_task = Task(
            description="""Translate the formatted English summary into Arabic, Hindi, and Hebrew.
            
            Requirements:
            - Maintain all Markdown formatting
            - Preserve image embeds and links
            - Use appropriate financial terminology for each language
            - Ensure cultural sensitivity and accuracy
            - Keep the same structure and length proportions
            
            Provide the output in this format:
            ENGLISH: [original formatted summary]
            ARABIC: [Arabic translation with formatting]
            HINDI: [Hindi translation with formatting]
            HEBREW: [Hebrew translation with formatting]""",
            expected_output="""A multilingual report containing:
            - Original English formatted summary
            - Arabic translation with formatting
            - Hindi translation with formatting
            - Hebrew translation with formatting""",
            agent=self.agents["translation_agent"],
            context=[formatting_task]
        )
        
        distribution_task = Task(
            description="""Prepare a detailed delivery report for the financial summaries.
            
            Process:
            1. Extract the English, Arabic, Hindi, and Hebrew summaries from the translation task output
            2. Format each language version with appropriate headers:
               - English: "🇺🇸 **US Financial Market Summary**"
               - Arabic: "🇸🇦 **ملخص السوق المالي الأمريكي**"
               - Hindi: "🇮🇳 **अमेरिकी वित्तीय बाजार सारांश**"
               - Hebrew: "🇮🇱 **סיכום שוק ההון האמריקאי**"
            3. Prepare the messages for Telegram delivery
            4. Provide a structured output with all language versions ready for sending
            
            Return the formatted messages in JSON format for processing.""",
            expected_output="JSON object with formatted messages for each language ready for Telegram delivery",
            agent=self.agents["distribution_agent"],
            context=[translation_task]
        )
        
        return [search_task, summary_task, formatting_task, translation_task, distribution_task]
    
    def _create_crew(self) -> Crew:
        """Create the CrewAI crew with sequential process"""
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
        
        logger.info("CrewAI crew created successfully")
        return crew
    
    async def execute_summary_workflow(self) -> Dict[str, Any]:
        """Execute the complete financial summary workflow"""
        try:
            logger.info("Starting financial summary workflow")
            start_time = datetime.now()
            
            # Execute CrewAI workflow
            result = self.crew.kickoff()
            
            # Process and send results via Telegram
            await self._process_and_send_results(result)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.info(f"Workflow completed successfully in {execution_time:.2f} seconds")
            return {
                "status": "success",
                "execution_time": execution_time,
                "timestamp": end_time.isoformat(),
                "result": str(result)
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _process_and_send_results(self, crew_result):
        """Process crew results and send via Telegram"""
        try:
            logger.info("Processing crew results for Telegram delivery...")
            
            # Extract task outputs
            if hasattr(crew_result, 'tasks_output') and crew_result.tasks_output:
                # Get translation task output (4th task, index 3)
                translation_output = crew_result.tasks_output[3]
                translation_text = str(translation_output.raw)
                
                logger.info("Parsing translation output...")
                
                # Extract language sections from the output
                sections = self._extract_language_sections(translation_text)
                
                # Send each language version to Telegram
                for lang_code, (header, content) in sections.items():
                    if content:
                        message = f"{header}\n\n{content}"
                        logger.info(f"Sending {lang_code} message to Telegram...")
                        result = await self.telegram_manager.send_message(message)
                        logger.info(f"Telegram result for {lang_code}: {result}")
                        await asyncio.sleep(2)  # Rate limiting
                
                logger.info("All messages sent successfully!")
                
        except Exception as e:
            logger.error(f"Failed to process and send results: {str(e)}")
            # Send error notification
            error_message = f"⚠️ **Error in Financial Summary**\n\nFailed to process results: {str(e)}"
            await self.telegram_manager.send_message(error_message)
    
    def _extract_language_sections(self, text: str) -> Dict[str, tuple]:
        """Extract language sections from translation output"""
        sections = {}
        
        # Define language patterns and headers
        patterns = {
            "english": {
                "pattern": r"ENGLISH:\s*(.*?)(?=ARABIC:|$)",
                "header": "🇺🇸 **US Financial Market Summary**"
            },
            "arabic": {
                "pattern": r"ARABIC:\s*(.*?)(?=HINDI:|$)", 
                "header": "🇸🇦 **ملخص السوق المالي الأمريكي**"
            },
            "hindi": {
                "pattern": r"HINDI:\s*(.*?)(?=HEBREW:|$)",
                "header": "🇮🇳 **अमेरिकी वित्तीय बाजार सारांश**"
            },
            "hebrew": {
                "pattern": r"HEBREW:\s*(.*?)$",
                "header": "🇮🇱 **סיכום שוק ההון האמריקאי**"
            }
        }
        
        import re
        for lang, config in patterns.items():
            match = re.search(config["pattern"], text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                # Clean up the content
                content = re.sub(r'^```\w*\n?', '', content)  # Remove code block markers
                content = re.sub(r'\n?```$', '', content)
                content = content.strip()
                sections[lang] = (config["header"], content)
        
        return sections
    
    def _parse_translation_output(self, text: str) -> Dict[str, str]:
        """Parse translation output from text format"""
        translations = {}
        
        # Extract each language section
        sections = {
            "english": r"ENGLISH:\s*(.*?)(?=ARABIC:|$)",
            "arabic": r"ARABIC:\s*(.*?)(?=HINDI:|$)",
            "hindi": r"HINDI:\s*(.*?)(?=HEBREW:|$)",
            "hebrew": r"HEBREW:\s*(.*?)$"
        }
        
        import re
        for lang, pattern in sections.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                translations[lang] = match.group(1).strip()
        
        return translations
    
    def _parse_formatting_output(self, text: str) -> Dict[str, List[str]]:
        """Parse formatting output from text format"""
        import re
        
        # Extract image URLs
        url_match = re.search(r"IMAGE_URLS:\s*\[(.*?)\]", text, re.DOTALL)
        urls = []
        if url_match:
            url_text = url_match.group(1)
            urls = [url.strip().strip('"\'') for url in url_text.split(',') if url.strip()]
        
        # Extract image descriptions
        desc_match = re.search(r"IMAGE_DESCRIPTIONS:\s*\[(.*?)\]", text, re.DOTALL)
        descriptions = []
        if desc_match:
            desc_text = desc_match.group(1)
            descriptions = [desc.strip().strip('"\'') for desc in desc_text.split(',') if desc.strip()]
        
        return {
            "image_urls": urls,
            "image_descriptions": descriptions
        }

class ScheduleManager:
    """Manages scheduling for daily execution"""
    
    def __init__(self, crew_system: FinancialSummaryCrewAI):
        self.crew_system = crew_system
        self.ist_timezone = pytz.timezone('Asia/Kolkata')
    
    def schedule_daily_execution(self):
        """Schedule daily execution at 01:30 IST"""
        schedule.every().day.at("01:30").do(self._run_scheduled_task)
        logger.info("Scheduled daily execution at 01:30 IST")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _run_scheduled_task(self):
        """Wrapper for running the async task in scheduler"""
        try:
            asyncio.run(self.crew_system.execute_summary_workflow())
        except Exception as e:
            logger.error(f"Scheduled task failed: {str(e)}")

def main():
    """Main function to run the financial summary system"""
    try:
        # Initialize the system
        crew_system = FinancialSummaryCrewAI()
        
        # Check if running in test mode
        if len(sys.argv) > 1 and sys.argv[1] == "--test":
            logger.info("Running in test mode - executing immediately")
            result = asyncio.run(crew_system.execute_summary_workflow())
            print(f"Test execution result: {result}")
        else:
            # Run scheduler
            logger.info("Starting scheduled execution mode")
            scheduler = ScheduleManager(crew_system)
            scheduler.schedule_daily_execution()
            
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()