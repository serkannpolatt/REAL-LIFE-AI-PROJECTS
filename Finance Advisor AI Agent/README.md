# Financial Advisor AI Chatbot

## Overview
This project is an AI-powered financial advisor chatbot that helps users analyze their investment portfolios and target allocations. It uses advanced language models to provide personalized financial advice based on uploaded data.

## Features
- Upload portfolio and target allocation CSV files
- Data preprocessing and cleaning
- AI chatbot powered by OpenAI GPT models
- Relevancy evaluation for chatbot responses
- User-friendly Streamlit web interface

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/finance-advisor-ai-chatbot.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```
2. Upload your portfolio and target allocation files (CSV format).
3. Enter your OpenAI API key in the sidebar.
4. Start chatting with the AI financial advisor!

## File Structure
- `main.py`: Streamlit app entry point
- `chatbot.py`: FinancialChatbot class (AI logic)
- `data_processing.py`: DataProcessor class (data cleaning)
- `requirements.txt`: Python dependencies
- `processed/`: Folder for processed CSV files

## Data Format
The chatbot uses two CSV files uploaded by the user:

### Target Allocations CSV
- **Client**: Identifier of the client
- **Target Portfolio**: Portfolio type (e.g., Balanced)
- **Asset Class**: Asset category (e.g., Bonds, ETFs, Stocks)
- **Target Allocation (%)**: Percentage allocation

### Financial Data CSV
- **Client**: Identifier of the client
- **Symbol**: Asset symbol
- **Name**: Asset name
- **Sector**: Asset sector
- **Quantity**: Number of units
- **Buy Price**: Purchase price
- **Current Price**: Current market price
- **Market Value**: Total market value
- **Purchase Date**: Date of purchase
- **Dividend Yield**: Dividend yield
- **P/E Ratio**: Price-to-earnings ratio
- **52-Week High/Low**: Highest/lowest price in last 52 weeks
- **Analyst Rating**: Analyst rating
- **Target Price**: Analyst target price
- **Risk Level**: Risk level

## Example Questions
- "What is the portfolio for Client 1?"
- "How much does Client 1 allocate to ETFs?"
- "What is the current market value of Amazon.com Inc. for Client 1?"
- "What are the target allocations for Client 2?"
- "What is the risk level of Client 1's holdings?"

## Tools Used
- [OpenAI](https://openai.com/) for GPT models
- [DeepEval](https://docs.confident-ai.com/) for evaluation
- [LangChain](https://python.langchain.com/) for LLM framework
- [Streamlit](https://streamlit.io/) for web interface
- [streamlit-chat](https://github.com/AI-Yash/st-chat) for chat UI