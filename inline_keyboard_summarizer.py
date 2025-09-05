#!/usr/bin/env python3


import os
import sys
import json
import asyncio
import logging
import schedule
import time
import pytz
import requests
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
from io import BytesIO
import re

# CrewAI imports
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import TavilySearchTool, SerperDevTool
from crewai.tools import BaseTool

# LiteLLM for model management
import litellm

# Image processing
from PIL import Image, ImageDraw, ImageFont
import aiohttp
from playwright.async_api import async_playwright

# Telegram imports
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
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
        self.model = "gemini/gemini-1.5-flash"
        
        # Configure LiteLLM globally for Gemini
        os.environ["GOOGLE_API_KEY"] = config.google_api_key
        os.environ["GEMINI_API_KEY"] = config.google_api_key
        litellm.set_verbose = False
        
    def get_llm_config(self):
        """Returns LLM configuration for CrewAI agents"""
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

class PlaceholderRemover:
    """Advanced placeholder text removal and content cleaning"""
    
    @staticmethod
    def remove_all_placeholders(text: str) -> str:
        """Remove all forms of placeholder text and clean content"""
        if not text:
            return ""
        
        # Remove image placeholders
        text = re.sub(r'\b[Ii]mage\s+\d+\b', '', text)
        text = re.sub(r'\b[Cc]hart\s+\d+\b', '', text)
        text = re.sub(r'\b[Gg]raph\s+\d+\b', '', text)
        text = re.sub(r'\b[Ff]igure\s+\d+\b', '', text)
        
        # Remove placeholder keywords
        placeholder_patterns = [
            r'\bplaceholder[_\s]*\w*\b',
            r'\b\[placeholder[^\]]*\]\b',
            r'\b\(placeholder[^)]*\)\b',
            r'\bTBD\b',
            r'\bTO BE DETERMINED\b',
            r'\bCOMING SOON\b',
            r'\bINSERT[_\s]*\w*\b'
        ]
        
        for pattern in placeholder_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove markdown image syntax
        text = re.sub(r'!\[[^\]]*\]\([^)]*placeholder[^)]*\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'!\[[^\]]*\]\(\s*\)', '', text)
        text = re.sub(r'!\[\s*\]\([^)]*\)', '', text)
        
        # Remove empty markdown links
        text = re.sub(r'\[[^\]]*\]\(\s*\)', '', text)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'\s{3,}', ' ', text)
        
        # Remove lines that are mostly placeholder text
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that are mostly placeholder indicators
            if re.search(r'^\s*[\[\(]?placeholder[\]\)]?\s*$', line, re.IGNORECASE):
                continue
            if re.search(r'^\s*image\s+\d+\s*$', line, re.IGNORECASE):
                continue
            if re.search(r'^\s*chart\s+\d+\s*$', line, re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines).strip()
        return result
    
    @staticmethod
    def enhance_content_without_placeholders(text: str) -> str:
        """Enhance content by replacing removed placeholders with meaningful descriptions"""
        
        # First remove all placeholders
        cleaned = PlaceholderRemover.remove_all_placeholders(text)
        
        # Add contextual analysis where placeholders were removed
        enhanced_sections = []
        
        # Look for sections that might have had images
        market_keywords = ['market overview', 's&p 500', 'nasdaq', 'dow jones', 'indices']
        sector_keywords = ['sector', 'performance', 'winners', 'losers', 'rotation']
        
        lines = cleaned.split('\n')
        current_section = []
        
        for line in lines:
            current_section.append(line)
            
            # Check if we should add analysis after this line
            line_lower = line.lower()
            
            # Add market analysis after market data
            if any(keyword in line_lower for keyword in market_keywords) and any(char.isdigit() for char in line):
                if not any('analysis' in l.lower() for l in current_section[-3:]):
                    analysis = PlaceholderRemover._generate_market_analysis_text()
                    current_section.append("")
                    current_section.append(analysis)
            
            # Add sector analysis after sector mentions
            elif any(keyword in line_lower for keyword in sector_keywords) and any(char.isdigit() for char in line):
                if not any('rotation' in l.lower() for l in current_section[-3:]):
                    analysis = PlaceholderRemover._generate_sector_analysis_text()
                    current_section.append("")
                    current_section.append(analysis)
        
        return '\n'.join(current_section).strip()
    
    @staticmethod
    def _generate_market_analysis_text() -> str:
        """Generate meaningful market analysis text"""
        return ("The current market dynamics reflect institutional trading patterns, "
                "with technical indicators showing momentum shifts across major indices. "
                "Volume analysis and price action suggest active participation from both "
                "retail and institutional investors responding to economic data and earnings results.")
    
    @staticmethod
    def _generate_sector_analysis_text() -> str:
        """Generate meaningful sector analysis text"""
        return ("Sector rotation patterns indicate shifting investor preferences based on "
                "economic outlook and interest rate expectations. Growth versus value dynamics "
                "continue to influence portfolio allocation decisions across institutional investors.")

class EnhancedTelegramManager:
    """Enhanced Telegram manager with image collages and inline keyboards"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.bot = Bot(token=config.telegram_bot_token)
        self.channel_id = config.telegram_channel_id
    
    async def send_message_with_chart_buttons(self, message: str, chart_urls: List[str], 
                                            chart_descriptions: List[str]) -> Dict[str, Any]:
        """Send message with inline keyboard buttons for charts"""
        try:
            keyboard = []
            
            for i, (url, desc) in enumerate(zip(chart_urls[:4], chart_descriptions[:4])):
                if self._is_valid_http_url(url):
                    # Using simple chart emoji
                    button_text = f"📊 {desc[:30]}" if desc else f"📊 Chart {i+1}"
                    keyboard.append([InlineKeyboardButton(button_text, url=url)])
            
            if len(chart_urls) > 1:
                keyboard.append([InlineKeyboardButton("📈 View All Charts", callback_data="all_charts")])
            
            reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
            
            await self.bot.send_message(
                chat_id=self.channel_id,
                text=message,
                reply_markup=reply_markup,
                parse_mode=None
            )
            
            logger.info(f"Message sent with {len(keyboard)} chart buttons")
            return {"status": "success", "message": "Message with chart buttons sent successfully"}
            
        except TelegramError as e:
            logger.error(f"Failed to send message with chart buttons: {str(e)}")
            return {"status": "error", "message": f"Failed to send message: {str(e)}"}
    
    async def create_and_send_chart_collage(self, chart_urls: List[str], chart_descriptions: List[str], 
                                          caption: str = "") -> Dict[str, Any]:
        """Download charts, create a collage, and send it"""
        try:
            chart_images = await self._download_chart_images(chart_urls[:4])
            
            if not chart_images:
                logger.warning("No chart images could be downloaded, creating informational image")
                collage = self._create_info_placeholder(chart_urls, chart_descriptions)
            else:
                collage = self._create_chart_collage(chart_images, chart_descriptions)
            
            # Convert to bytes for sending
            img_bytes = BytesIO()
            collage.save(img_bytes, format='PNG', optimize=True, quality=85)
            img_bytes.seek(0)
            
            # Send the collage
            await self.bot.send_photo(
                chat_id=self.channel_id,
                photo=img_bytes,
                caption=caption[:1024] if caption else "📊 Financial Market Charts Overview"
            )
            
            logger.info("Chart collage sent successfully")
            return {"status": "success", "message": "Chart collage sent successfully"}
            
        except Exception as e:
            logger.error(f"Failed to create/send chart collage: {str(e)}")
            return {"status": "error", "message": f"Failed to send collage: {str(e)}"}
    
    async def _download_chart_images(self, urls: List[str]) -> List[Tuple[Image.Image, str]]:
        """Download and process chart images with screenshot capability"""
        images = []

        async with aiohttp.ClientSession() as session:
            for i, url in enumerate(urls):
                try:
                    if not self._is_valid_http_url(url):
                        continue

                    if self._looks_like_image_url(url):
                        img = await self._download_direct_image(session, url)
                        if img:
                            images.append((img, f"Chart {i+1}"))
                    else:
                        # Try to take screenshot of the chart page
                        screenshot_img = await self._take_chart_screenshot(url)
                        if screenshot_img:
                            images.append((screenshot_img, f"Chart {i+1}"))
                        else:
                            # Fallback to placeholder if screenshot fails
                            placeholder = self._create_chart_placeholder(url, f"Chart {i+1}")
                            if placeholder:
                                images.append((placeholder, f"Chart {i+1}"))

                except Exception as e:
                    logger.warning(f"Failed to process chart {i+1} from {url}: {str(e)}")
                    continue

        return images
    
    async def _download_direct_image(self, session: aiohttp.ClientSession, url: str) -> Optional[Image.Image]:
        """Download a direct image URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    content = await response.read()
                    img = Image.open(BytesIO(content))
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    return img
        except Exception as e:
            logger.warning(f"Failed to download image from {url}: {str(e)}")
        
        return None

    async def _take_chart_screenshot(self, url: str) -> Optional[Image.Image]:
        """Take a screenshot of a chart page using Playwright"""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()

                # Set viewport size for better screenshot
                await page.set_viewport_size({"width": 1200, "height": 800})

                # Navigate to the URL
                await page.goto(url, wait_until="networkidle")

                # Wait a bit for dynamic content to load
                await asyncio.sleep(2)

                # Take screenshot
                screenshot_bytes = await page.screenshot(full_page=False)

                # Close browser
                await browser.close()

                # Convert to PIL Image
                img = Image.open(BytesIO(screenshot_bytes))
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')

                logger.info(f"Successfully took screenshot of {url}")
                return img

        except Exception as e:
            logger.warning(f"Failed to take screenshot of {url}: {str(e)}")
            return None

    def _create_chart_placeholder(self, url: str, title: str) -> Optional[Image.Image]:
        """Create a placeholder image with chart link information"""
        try:
            img = Image.new('RGB', (400, 300), color='#f8f9fa')
            draw = ImageDraw.Draw(img)
            
            try:
                font_title = ImageFont.truetype("arial.ttf", 20)
                font_text = ImageFont.truetype("arial.ttf", 14)
            except:
                font_title = ImageFont.load_default()
                font_text = ImageFont.load_default()
            
            # Draw chart representation
            draw.rectangle([50, 80, 350, 200], outline='#007bff', width=3)
            draw.line([80, 170, 120, 140, 160, 160, 200, 120, 240, 150, 280, 110, 320, 140], 
                     fill='#007bff', width=3)
            
            # Add title
            title_bbox = draw.textbbox((0, 0), title, font=font_title)
            title_width = title_bbox[2] - title_bbox[0]
            draw.text(((400 - title_width) // 2, 30), title, fill='#333333', font=font_title)
            
            # Add "Interactive Chart Available" text
            view_text = "Interactive Chart Available"
            text_bbox = draw.textbbox((0, 0), view_text, font=font_text)
            text_width = text_bbox[2] - text_bbox[0]
            draw.text(((400 - text_width) // 2, 220), view_text, fill='#666666', font=font_text)
            
            # Add domain name
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                domain_text = f"Source: {domain}"
                domain_bbox = draw.textbbox((0, 0), domain_text, font=font_text)
                domain_width = domain_bbox[2] - domain_bbox[0]
                draw.text(((400 - domain_width) // 2, 250), domain_text, fill='#999999', font=font_text)
            except:
                pass
            
            return img
            
        except Exception as e:
            logger.warning(f"Failed to create placeholder for {url}: {str(e)}")
            return None
    
    def _create_info_placeholder(self, urls: List[str], descriptions: List[str]) -> Image.Image:
        """Create an informational image when no charts are available"""
        img = Image.new('RGB', (800, 600), color='#f8f9fa')
        draw = ImageDraw.Draw(img)
        
        try:
            font_title = ImageFont.truetype("arial.ttf", 24)
            font_text = ImageFont.truetype("arial.ttf", 16)
        except:
            font_title = ImageFont.load_default()
            font_text = ImageFont.load_default()
        
        # Title
        title = "Financial Market Charts"
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(((800 - title_width) // 2, 50), title, fill='#333333', font=font_title)
        
        # Subtitle
        subtitle = "Interactive charts available via buttons above"
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=font_text)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        draw.text(((800 - subtitle_width) // 2, 100), subtitle, fill='#666666', font=font_text)
        
        # List available charts
        y_offset = 150
        for i, (url, desc) in enumerate(zip(urls[:4], descriptions[:4])):
            if url:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    chart_text = f"{i+1}. {desc if desc else 'Market Chart'} ({domain})"
                except:
                    chart_text = f"{i+1}. {desc if desc else 'Market Chart'}"
                
                draw.text((50, y_offset + i * 40), chart_text, fill='#333333', font=font_text)
        
        return img
    
    def _create_chart_collage(self, chart_images: List[Tuple[Image.Image, str]], 
                            descriptions: List[str]) -> Image.Image:
        """Create a collage from multiple chart images"""
        if not chart_images:
            return Image.new('RGB', (800, 600), color='#f8f9fa')
        
        num_images = len(chart_images)
        if num_images == 1:
            cols, rows = 1, 1
            collage_size = (800, 600)
        elif num_images == 2:
            cols, rows = 2, 1
            collage_size = (1000, 500)
        elif num_images <= 4:
            cols, rows = 2, 2
            collage_size = (1000, 800)
        else:
            cols, rows = 3, 2
            collage_size = (1200, 800)
        
        collage = Image.new('RGB', collage_size, color='#ffffff')
        
        img_width = collage_size[0] // cols
        img_height = collage_size[1] // rows
        
        for i, (img, title) in enumerate(chart_images[:cols*rows]):
            if i >= cols * rows:
                break
                
            row = i // cols
            col = i % cols
            x = col * img_width
            y = row * img_height
            
            img_resized = img.resize((img_width - 10, img_height - 40), Image.Resampling.LANCZOS)
            collage.paste(img_resized, (x + 5, y + 5))
            
            # Add title overlay
            draw = ImageDraw.Draw(collage)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            title_text = descriptions[i] if i < len(descriptions) and descriptions[i] else title
            title_text = title_text[:30] + "..." if len(title_text) > 30 else title_text
            
            # Background for title
            draw.rectangle([x, y + img_height - 35, x + img_width, y + img_height], 
                         fill='#000000')
            
            # Title text
            text_bbox = draw.textbbox((0, 0), title_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            draw.text((x + (img_width - text_width) // 2, y + img_height - 30), 
                     title_text, fill='white', font=font)
        
        return collage
    
    async def send_contextual_charts(self, message: str, chart_urls: List[str], 
                                   chart_descriptions: List[str]) -> Dict[str, Any]:
        """Send message with contextually placed chart buttons and collages"""
        try:
            # Clean message of placeholders
            clean_message = PlaceholderRemover.enhance_content_without_placeholders(message)
            
            # Create inline keyboard for charts
            keyboard = []
            for i, (url, desc) in enumerate(zip(chart_urls[:3], chart_descriptions[:3])):
                if self._is_valid_http_url(url):
                    button_text = f"📊 {desc[:25]}" if desc else f"📊 Chart {i+1}"
                    keyboard.append([InlineKeyboardButton(button_text, url=url)])
            
            keyboard.append([InlineKeyboardButton("🖼️ View Charts Collage", callback_data="show_collage")])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Send main message with buttons
            await self.bot.send_message(
                chat_id=self.channel_id,
                text=clean_message,
                reply_markup=reply_markup,
                parse_mode=None
            )
            
            await asyncio.sleep(2)
            
            # Create and send chart collage
            await self.create_and_send_chart_collage(
                chart_urls, 
                chart_descriptions,
                "Today's Financial Market Performance"
            )
            
            return {"status": "success", "message": "Contextual charts sent successfully"}
            
        except Exception as e:
            logger.error(f"Failed to send contextual charts: {str(e)}")
            return {"status": "error", "message": f"Failed to send contextual charts: {str(e)}"}
    
    def _is_valid_http_url(self, url: str) -> bool:
        """Check if URL is valid HTTP/HTTPS"""
        try:
            return url.startswith("http://") or url.startswith("https://")
        except:
            return False
    
    def _looks_like_image_url(self, url: str) -> bool:
        """Check if URL looks like a direct image"""
        return url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))

class FinancialSummaryCrewAI:
    """Main CrewAI system for financial market summaries with enhanced image handling"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.llm_manager = LLMManager(self.config)
        self.search_manager = SearchToolManager(self.config)
        self.telegram_manager = EnhancedTelegramManager(self.config)
        
        self.agents = self._create_agents()
        self.tasks = self._create_tasks()
        self.crew = self._create_crew()
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create all CrewAI agents"""
        llm_config = self.llm_manager.get_llm_config()
        
        agents = {
            "search_agent": Agent(
                role="Financial News Researcher",
                goal="Search for the latest US financial market news and data from the past 2 hours",
                backstory="""You are an expert financial researcher with deep knowledge of US markets.
                You specialize in finding the most relevant and timely financial news, market movements,
                and trading activity. You focus on REAL data and avoid placeholder information.""",
                tools=[self.search_manager.search_tool],
                llm=llm_config,
                verbose=True,
                allow_delegation=False,
                max_iter=3
            ),
            
            "summary_agent": Agent(
                role="Financial News Summarizer",
                goal="Create detailed, informative summaries with REAL market data only",
                backstory="""You are a skilled financial journalist who excels at distilling complex
                market information into clear, actionable summaries. You NEVER use placeholder text
                and only include actual market data with specific numbers and percentages.""",
                llm=llm_config,
                verbose=True,
                allow_delegation=False,
                max_iter=2
            ),
            
            "formatting_agent": Agent(
                role="Content Formatter and Chart Finder",
                goal="Format content cleanly and find relevant financial chart URLs",
                backstory="""You are a content specialist who formats financial summaries for
                digital platforms and finds relevant chart URLs. You completely avoid placeholder
                text and focus on clean, professional presentation.""",
                tools=[self.search_manager.search_tool],
                llm=llm_config,
                verbose=True,
                allow_delegation=False,
                max_iter=2
            ),
            
            "translation_agent": Agent(
                role="Multilingual Financial Translator",
                goal="Translate financial content into Arabic, Hindi, and Hebrew without placeholders",
                backstory="""You are an expert translator specializing in financial terminology.
                You maintain accuracy while ensuring NO placeholder text appears in any translation.""",
                llm=llm_config,
                verbose=True,
                allow_delegation=False,
                max_iter=2
            ),
            
            "distribution_agent": Agent(
                role="Content Distribution Specialist",
                goal="Prepare clean content for Telegram distribution",
                backstory="""You ensure all content is properly formatted for Telegram delivery
                and completely free of placeholder text or broken formatting.""",
                llm=llm_config,
                verbose=True,
                allow_delegation=False,
                max_iter=2
            )
        }
        
        logger.info("All agents created successfully")
        return agents
    
    def _create_tasks(self) -> List[Task]:
        """Create all CrewAI tasks with strict no-placeholder requirements"""
        
        search_task = Task(
            description="""Search for REAL-TIME US financial market data from the last 2 hours:
            - Current S&P 500, NASDAQ, Dow Jones values and percentage changes
            - Latest earnings reports and corporate announcements
            - Federal Reserve updates and economic policy news
            - Currency rates, commodity prices, and trading volumes
            - Breaking financial news with actual impact on markets
            
            CRITICAL: Provide ONLY verified, current market data with specific numbers.
            If current data is unavailable, clearly state this limitation.
            NEVER use placeholder text or generic descriptions.""",
            expected_output="""Detailed report with:
            - Specific market data with actual numbers and percentages
            - News items with verified sources and timestamps  
            - Clear indication if real-time data is limited
            - NO placeholder text or generic statements""",
            agent=self.agents["search_agent"]
        )
        
        summary_task = Task(
            description="""Create a comprehensive 400-500 word summary using ONLY real market data:
            
            Structure:
            1. Market Overview (actual index values and changes)
            2. Key News Highlights (specific, verified events)
            3. Sector Performance (real percentages and movements)
            4. Daily Market Analysis (detailed trading insights)
            5. Market Outlook (based on actual data trends)
            
            STRICT REQUIREMENTS:
            - Use ONLY verified market data with specific numbers
            - NO placeholder text, generic statements, or "TBD" content
            - Include actual percentages, dollar amounts, and index levels
            - Reference real news sources and specific events
            - If data is limited, provide meaningful analysis of available information
            - Every section must contain substantive, real information""",
            expected_output="Professional 400-500 word summary with verified market data and zero placeholder content",
            agent=self.agents["summary_agent"],
            context=[search_task]
        )
        
        formatting_task = Task(
            description="""Format the summary and find 2-3 relevant financial chart URLs:
            
            Formatting Tasks:
            1. Clean the summary text for Telegram (remove excessive markdown)
            2. Search for legitimate financial chart URLs from reputable sources:
               - finance.yahoo.com chart URLs
               - finviz.com sector maps  
               - tradingview.com charts
               - marketwatch.com visualizations
            3. Find REAL chart URLs (not placeholder links)
            4. Ensure content flows naturally without placeholder gaps
            
            CRITICAL REQUIREMENTS:
            - Find actual, working chart URLs from financial websites
            - Remove any placeholder text completely
            - Ensure smooth content flow
            - Verify all URLs are from legitimate financial sources
            
            Output format:
            CLEANED_SUMMARY: [Complete summary with no placeholders]
            CHART_URLS: [url1, url2, url3]
            CHART_DESCRIPTIONS: [Market Overview Chart, Sector Performance, Index Comparison]""",
            expected_output="Formatted content with verified chart URLs and completely cleaned summary text",
            agent=self.agents["formatting_agent"],
            context=[summary_task]
        )
        
        translation_task = Task(
            description="""Translate the cleaned summary into Arabic, Hindi, and Hebrew:
            
            Requirements:
            - Maintain professional financial terminology
            - Ensure cultural appropriateness
            - Keep the same informational structure
            - Completely eliminate any placeholder text in all languages
            - Provide rich, detailed market analysis instead of placeholder content
            
            Format:
            ENGLISH: [original cleaned summary]
            ARABIC: [complete Arabic translation with detailed market analysis]
            HINDI: [complete Hindi translation with detailed market analysis] 
            HEBREW: [complete Hebrew translation with detailed market analysis]""",
            expected_output="Complete multilingual content with detailed market analysis in all languages",
            agent=self.agents["translation_agent"],
            context=[formatting_task]
        )
        
        distribution_task = Task(
            description="""Prepare final content for Telegram delivery:
            
            Process:
            1. Extract all language versions
            2. Add appropriate headers for each language
            3. Ensure all content is clean and professional
            4. Verify complete removal of placeholder text
            5. Format for optimal Telegram presentation
            
            Return structured JSON with all content ready for enhanced delivery.""",
            expected_output="JSON with cleaned, formatted messages ready for enhanced Telegram delivery",
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
            logger.info("Starting enhanced financial summary workflow")
            start_time = datetime.now()
            
            result = self.crew.kickoff()
            await self._process_and_send_enhanced_results(result)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.info(f"Enhanced workflow completed in {execution_time:.2f} seconds")
            return {
                "status": "success",
                "execution_time": execution_time,
                "timestamp": end_time.isoformat(),
                "result": str(result)
            }
            
        except Exception as e:
            logger.error(f"Enhanced workflow execution failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _process_and_send_enhanced_results(self, crew_result):
        """Process crew results with enhanced image handling"""
        try:
            logger.info("Processing crew results with enhanced image handling...")
            
            if hasattr(crew_result, 'tasks_output') and crew_result.tasks_output:
                formatting_output = crew_result.tasks_output[2] if len(crew_result.tasks_output) > 2 else None
                translation_output = crew_result.tasks_output[3] if len(crew_result.tasks_output) > 3 else None
                
                formatting_text = str(formatting_output.raw) if formatting_output else ""
                translation_text = str(translation_output.raw) if translation_output else ""
                
                # Extract cleaned summary and chart URLs
                cleaned_summary = self._extract_cleaned_summary(formatting_text)
                chart_urls = self._extract_chart_urls(formatting_text)
                chart_descriptions = self._extract_chart_descriptions(formatting_text)
                
                # Ensure we have some chart URLs
                if len(chart_urls) < 2:
                    chart_urls.extend(self._get_fallback_chart_urls()[:2-len(chart_urls)])
                
                # Parse language sections
                language_sections = self._extract_language_sections(translation_text)
                
                # Send enhanced content for each language
                for lang_code, (header, content) in language_sections.items():
                    if content:
                        # Apply final placeholder removal
                        final_content = PlaceholderRemover.enhance_content_without_placeholders(content)
                        
                        if lang_code == "english":
                            # Send English version with enhanced chart integration
                            await self.telegram_manager.send_contextual_charts(
                                f"{header}\n\n{final_content}",
                                chart_urls,
                                chart_descriptions
                            )
                        else:
                            # Send other languages as clean text
                            clean_message = f"{header}\n\n{final_content}"
                            await self.telegram_manager.send_message_with_chart_buttons(
                                clean_message, chart_urls, chart_descriptions
                            )
                        
                        await asyncio.sleep(3)
                
                logger.info("Enhanced results sent successfully!")
                
        except Exception as e:
            logger.error(f"Failed to process enhanced results: {str(e)}")
            error_message = f"Warning: Error in Enhanced Financial Summary\n\nProcessing failed: {str(e)}"
            await self.telegram_manager.bot.send_message(
                chat_id=self.telegram_manager.channel_id,
                text=error_message
            )
    
    def _extract_cleaned_summary(self, text: str) -> str:
        """Extract cleaned summary from formatting output"""
        match = re.search(r"CLEANED_SUMMARY:\s*(.*?)(?=CHART_URLS:|$)", text, re.DOTALL | re.IGNORECASE)
        if match:
            summary = match.group(1).strip()
            return PlaceholderRemover.remove_all_placeholders(summary)
        return PlaceholderRemover.remove_all_placeholders(text)
    
    def _extract_chart_urls(self, text: str) -> List[str]:
        """Extract chart URLs from formatting output"""
        match = re.search(r"CHART_URLS:\s*\[(.*?)\]", text, re.DOTALL)
        if match:
            urls_text = match.group(1)
            urls = [url.strip().strip('"\'') for url in urls_text.split(',') if url.strip()]
            return [url for url in urls if self._is_valid_chart_url(url)]
        return []
    
    def _extract_chart_descriptions(self, text: str) -> List[str]:
        """Extract chart descriptions from formatting output"""
        match = re.search(r"CHART_DESCRIPTIONS:\s*\[(.*?)\]", text, re.DOTALL)
        if match:
            desc_text = match.group(1)
            descriptions = [desc.strip().strip('"\'') for desc in desc_text.split(',') if desc.strip()]
            # Clean any placeholder descriptions
            cleaned_descriptions = []
            for desc in descriptions:
                if not re.search(r'placeholder|tbd|chart\s*\d+', desc, re.IGNORECASE):
                    cleaned_descriptions.append(desc)
                else:
                    cleaned_descriptions.append("Market Analysis Chart")
            return cleaned_descriptions
        return ["Market Performance", "Sector Analysis", "Trading Overview"]
    
    def _get_fallback_chart_urls(self) -> List[str]:
        """Get reliable fallback chart URLs"""
        return [
            "https://finance.yahoo.com/quote/%5EGSPC/",
            "https://finviz.com/map.ashx?t=sec_all",
            "https://finance.yahoo.com/quote/%5EIXIC/",
            "https://www.tradingview.com/symbols/SPX/"
        ]
    
    def _is_valid_chart_url(self, url: str) -> bool:
        """Validate that URL is from a legitimate financial source"""
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            return False
        
        legitimate_domains = [
            'finance.yahoo.com', 'finviz.com', 'tradingview.com', 
            'marketwatch.com', 'investing.com', 'bloomberg.com'
        ]
        
        return any(domain in url.lower() for domain in legitimate_domains)
    
    def _extract_language_sections(self, text: str) -> Dict[str, tuple]:
        """Extract language sections from translation output"""
        sections = {}
        
        patterns = {
            "english": {
                "pattern": r"ENGLISH:\s*(.*?)(?=ARABIC:|$)",
                "header": "US Financial Market Daily Summary"
            },
            "arabic": {
                "pattern": r"ARABIC:\s*(.*?)(?=HINDI:|$)", 
                "header": "ملخص السوق المالي الأمريكي اليومي"
            },
            "hindi": {
                "pattern": r"HINDI:\s*(.*?)(?=HEBREW:|$)",
                "header": "अमेरिकी वित्तीय बाजार दैनिक सारांश"
            },
            "hebrew": {
                "pattern": r"HEBREW:\s*(.*?)$",
                "header": "סיכום יומי של שוק הון האמריקאי"
            }
        }
        
        for lang, config in patterns.items():
            match = re.search(config["pattern"], text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
                content = PlaceholderRemover.remove_all_placeholders(content)
                sections[lang] = (config["header"], content)
        
        return sections

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
            time.sleep(60)
    
    def _run_scheduled_task(self):
        """Wrapper for running the async task in scheduler"""
        try:
            asyncio.run(self.crew_system.execute_summary_workflow())
        except Exception as e:
            logger.error(f"Scheduled task failed: {str(e)}")

def main():
    """Main function to run the enhanced financial summary system"""
    try:
        crew_system = FinancialSummaryCrewAI()
        
        if len(sys.argv) > 1 and sys.argv[1] == "--test":
            logger.info("Running enhanced test mode")
            result = asyncio.run(crew_system.execute_summary_workflow())
            print(f"Enhanced test execution result: {result}")
        else:
            logger.info("Starting enhanced scheduled execution mode")
            scheduler = ScheduleManager(crew_system)
            scheduler.schedule_daily_execution()
            
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
    except Exception as e:
        logger.error(f"Enhanced system initialization failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":

    main()
