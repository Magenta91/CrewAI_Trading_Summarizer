# Financial Market Summary Bot 📈

A comprehensive CrewAI-powered system that generates daily US financial market summaries and distributes them via Telegram in multiple languages (English, Arabic, Hindi, Hebrew).

## Features

- **Multi-Agent System**: Uses CrewAI framework with specialized agents for search, summarization, formatting, translation, and distribution
- **Intelligent Search**: Leverages Tavily or Serper APIs for real-time financial news
- **LLM Integration**: Uses Groq's free tier (llama3-8b-8192) via LiteLLM
- **Multi-Language Support**: Automatic translation to Arabic, Hindi, and Hebrew
- **Visual Enhancement**: Automatically finds and embeds relevant financial charts
- **Scheduled Execution**: Runs daily at 01:30 IST (after US market close)
- **Telegram Distribution**: Sends formatted summaries to Telegram channels
- **Comprehensive Logging**: Full execution logging and error handling

## Quick Start

### 1. Installation

```bash
# Clone or download the project files
# Ensure Python 3.10+ is installed

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the `.env` file and configure your API keys:

```bash
# Required API Keys:
GOOGLE_API_KEY=your_google_api_key      # Free from https://makersuite.google.com/app/apikey
TAVILY_API_KEY=your_tavily_api_key      # From https://tavily.com/
TELEGRAM_BOT_TOKEN=your_bot_token       # From @BotFather on Telegram
TELEGRAM_CHANNEL_ID=your_channel_id     # Your Telegram channel ID
```

### 3. API Key Setup

#### Google Gemini API (Free Tier)
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign up for a free account
3. Generate an API key
4. Add to `.env` file

#### Tavily Search API
1. Visit [Tavily](https://tavily.com/)
2. Sign up and get your API key
3. Add to `.env` file

#### Telegram Bot
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Create a new bot with `/newbot`
3. Get your bot token
4. Add your bot to your channel as an admin
5. Get your channel ID (use @userinfobot or check channel info)

### 4. Running the System

#### Test Mode (Immediate Execution)
```bash
python financial_market_summary.py --test
# or
python test_summary.py
```

#### Scheduled Mode (Daily at 01:30 IST)
```bash
python financial_market_summary.py
```

## System Architecture

### Agents

1. **Search Agent**: Finds latest US financial news using Tavily/Serper
2. **Summary Agent**: Creates concise 400-500 word summaries
3. **Formatting Agent**: Adds visual elements and proper formatting
4. **Translation Agent**: Translates to Arabic, Hindi, and Hebrew
5. **Distribution Agent**: Sends content via Telegram

### Workflow

```
Search → Summarize → Format → Translate → Distribute
```

Each step includes validation and error handling with detailed logging.

## File Structure

```
├── financial_market_summary.py  # Main CrewAI system
├── test_summary.py             # Test script
├── requirements.txt            # Python dependencies
├── .env                       # Configuration file
├── README.md                  # This file
└── financial_summary.log      # Generated log file
```

## Configuration Options

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google Gemini API key for LLM |
| `TAVILY_API_KEY` | Yes* | Tavily search API key |
| `SERPER_API_KEY` | Yes* | Alternative to Tavily |
| `TELEGRAM_BOT_TOKEN` | Yes | Telegram bot token |
| `TELEGRAM_CHANNEL_ID` | Yes | Target channel ID |

*At least one search API key is required

### Scheduling

The system runs daily at **01:30 IST** (after US market close at 4:00 PM EST).

To change the schedule, modify the time in `ScheduleManager.schedule_daily_execution()`:

```python
schedule.every().day.at("01:30").do(self._run_scheduled_task)
```

## Output Format

### English Summary Structure
- Market Overview (major indices)
- Key News Highlights (2-3 stories)
- Sector Performance
- Looking Ahead

### Multi-Language Output
- 🇺🇸 English (original)
- 🇸🇦 Arabic (العربية)
- 🇮🇳 Hindi (हिंदी)
- 🇮🇱 Hebrew (עברית)

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify all API keys in `.env` file
   - Check API key permissions and quotas

2. **Telegram Errors**
   - Ensure bot is admin in the target channel
   - Verify channel ID format (should start with `-` for channels)

3. **Search Errors**
   - Check internet connection
   - Verify search API quotas

4. **Translation Issues**
   - LLM may occasionally fail translations
   - System will retry with error handling

### Logs

Check `financial_summary.log` for detailed execution logs and error messages.

### Testing

Run the test script to verify system functionality:

```bash
python test_summary.py
```

## Customization

### Adding New Languages
Modify the translation task in `_create_tasks()` method to include additional languages.

### Changing Summary Length
Adjust the word count in the summary task description (currently 400-500 words).

### Custom Scheduling
Modify the `ScheduleManager` class to change execution timing or frequency.

## Dependencies

- `crewai==0.51.1` - Multi-agent framework
- `litellm==1.40.14` - LLM integration
- `crewai-tools==0.8.1` - Search tools
- `python-telegram-bot==21.4` - Telegram integration
- `python-dotenv==1.0.1` - Environment management
- `schedule==1.2.1` - Task scheduling
- `pytz==2024.1` - Timezone handling

## License

This project is provided as-is for educational and personal use.

## Support

For issues or questions:
1. Check the logs in `financial_summary.log`
2. Verify your API keys and configuration
3. Test with `test_summary.py` first
4. Review the troubleshooting section above